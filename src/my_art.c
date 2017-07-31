#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <assert.h>
#include "art.h"

/**Getting intrinsics for vector comparison */
#ifdef __i386__
    #include <emmintrin.h>
#else 
#ifdef __amd64__
    #include <emmintrin.h>
#endif 
#endif 

/** defining c++ delete as zfree
 * Copied from Linus Torvalds linux/tools/perf/util/util.h 
 */
#ifndef zfree
# define zfree(ptr) ({ free(*ptr); *ptr = NULL; })
#endif


/** We are manipulating pointertags to store information in accordance with
 *  the paper first decribing the datastructure and the authors c++ 
 *  implementation.
 *  
 *  Here are some makros that help with that pointer maintanance*/

#define IS_LEAF(x) (((uintptr_t)x & 1))
#define SET_LEAF(x) ((void*)((uintptr_t)x | 1))
#define LEAF_RAW(x) ((art_leaf*)((void*)((uintptr_t)x & ~1)))

/**
 * using libarts compact way to allocate memory for all the the tree nodes.
 * It uses the type and initializes the corresponding node struct to 0
 * Rewrote it slightly to use a passthrough node reference instead of
 * possibly casting Null to an art_node and then setting the type attribute
 * on the null pointer.  I dont know if this was safe, but i didn't like it. 
 * 
 * @return:  0 , in case of memory allocation error
 *          -1 , if memory allocation fails  */

static int alloc_node(uint8_t type,art_node *node) {
    switch (type) { 
        case NODE4:
            node = (art_node*)calloc(1, sizeof(art_node4));
            break;
        case NODE16:
            node = (art_node*)calloc(1, sizeof(art_node16));
            break;
        case NODE48:
            node = (art_node*)calloc(1, sizeof(art_node48));
            break;
        case NODE256:
            node = (art_node*)calloc(1, sizeof(art_node256));
            break;
        default:
            abort();
    }
    if(node){
        node->type = type;
        return 0;
    }
    else
        return -1; 
}

int art_tree_init(art_tree *t) {
    t->root = NULL;
    t->size = 0;
    return 0;
}


/*Destruction of tree using recursion*/
static void destroy_node(art_node *node) {
    /* Break if null */
    if (!node) return;
    
    /* Special case leafs */
    if (IS_LEAF(node)) {
       art_leaf *leaf = LEAF_RAW(node);
       zfree(&leaf);
       return;
    }

    /* Handle each node type */
    int i;
    union { 
        art_node4 *p1;
        art_node16 *p2;
        art_node48 *p3;
        art_node256 *p4;
    } node_ptrs;
    switch (node->type) { 
        case NODE4:
            node_ptrs.p1 = (art_node4*)node;
            for (i=0; i < node->num_children; i++){
                destroy_node(node_ptrs.p1->children[i]);
            }
            break;
           
        case NODE16:
            node_ptrs.p2 = (art_node16*)node;
            for (i=0; i< node->num_children; i++) {
                destroy_node(node_ptrs.p2->children[i]);
            }
            break;
         
        case NODE48:
            node_ptrs.p3 = (art_node48*)node;
            for (i=0; i< node->num_children; i++) {
                destroy_node(node_ptrs.p3->children[i]);
            }
            break;
        /* Slightly different because children are handled 
         * differently in 256 nodes*/
        case NODE256:
            node_ptrs.p4 = (art_node256*)node;
            for (i=0; i<256; i++) {
                if (node_ptrs.p4->children[i])
                    destroy_node(node_ptrs.p4->children[i]);
            }
            break;
        
        default:
            abort();
    }
    /* When all children are handled free yourself */
    zfree(&node);
}

/**
 * Destruction of ART tree
 * @return 0 on success 
 */
int art_tree_destroy(art_tree *t) {
    destroy_node(t->root);
    return 0;
}

/**
 * Size of ART tree
 * Uses inline art_size if possible
 */

#ifndef BROKEN_GCC_C99_INLINE
extern inline uint64_t art_size(art_tree *t);
#endif

static art_node** find_child(art_node *node, unsigned char c) {
    int i, mask, bitfield;
    union {
        art_node4 *p1;
        art_node16 *p2;
        art_node48 *p3;
        art_node256 *p4;
    } n_ptrs;
    switch (node->type) {
        case NODE4:
            n_ptrs.p1 = (art_node4*)node;
            for (i=0; i < node->num_children; i++) { 
            /* According Riley Berton, the following cast is a workaround
             * for a bug in gcc 5.1 when unrolling loops.
             * My target system uses 4.9 and there might be a chance that an 
             * upgrade will trigger this bug otherwise.
             *
             * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=59124
             */    
                if(((unsigned char*)n_ptrs.p1->keys)[i] == c)
                    return &n_ptrs.p1->children[i];
            }
            break;
        /* Now comes the Node16 operation which uses cpu vector magic
         * inorder to efficiently check all children at once
         * It gets it own scope in the original project and I do the same 
         */
        {
            case NODE16:
                n_ptrs.p2 = (art_node16*)node;
                // non-x86
                #ifdef __i386__
                    // Compare the key to all 16 using the intrinsic compare
                    __m128i cmp;
                    cmp = _mm_cmpeq_epi8(_mm_set1_epi8(c),
                            /* Load unalined 128Bytes of goodness 
                             * from keys atr */
                            _mm_loadu_si128((__m128i*)n_ptrs.p2->keys));
                    
                    /* Using mask to ignore children that don't exist
                     * Bitshift*n, subtract one, get n-sized mask, profit */
                    mask = (1 << node->num_children) - 1;
                    bitfield = _mm_movemask_epi8(cmp) & mask;
                #else
                #ifdef __amd64__
                    // Same as above
                    __m128i cmp;
                    cmp = _mm_cmpeq_epi8( _mm_set1_epi8(c),
                            _mm_loadu_si128(( __m128i*)n_ptrs.p2->keys));
                    mask = (1 << node->num_children) -1;
                    bitfield &= _mm_movemask_epi8(cmp) & mask;
                #else
                    /* If we are on some other architecture no builtins 
                     * Todo: Check if this works*/
                    bitfield = 0;
                    for (i = 0; i < 16; ++i) {
                        if (n_ptrs.p2->keys[i] == c)
                            bitfield |= (1 << i);
                    }
                    mask = (1 << node->num_children) -1;
                    bitfield &= mask; 
                #endif
                #endif
                
                /*
                 * If I have understood the datastructure correctly we should 
                 * only have one match and can get the index by counting the 
                 * trailing zeros. 
                 * Errorhandling could be done with the leading zeros
                 */
                 if (bitfield)
                     return &n_ptrs.p2->children[__builtin_ctz(bitfield)];
                 break;
        }
        
        case NODE48:
            n_ptrs.p3 = (art_node48*)node;
            i = n_ptrs.p3->keys[c];
            if (i)
                return &n_ptrs.p3->children[i-1];
            break;

        case NODE256:
            n_ptrs.p4 = (art_node256*)node;
            if (n_ptrs.p4->children[c])
                return &n_ptrs.p4->children[c];
            break;
        
        default:
            abort();
    }
    return NULL;
}
          
            



