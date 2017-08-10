#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <assert.h>
#include "my_art.h"

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


/* 
 * We are manipulating pointertags to store information in accordance with
 * the paper first decribing the datastructure and the authors c++ 
 * implementation.
 *  
 *  Here are some makros that help with that pointer maintanance
 */

#define IS_LEAF(x) (((uintptr_t)x & 1))
#define SET_LEAF(x) ((void*)((uintptr_t)x | 1))
#define LEAF_RAW(x) ((art_leaf*)((void*)((uintptr_t)x & ~1)))

/**
 * allocate memory for all the the tree node types.
 *
 * It uses the type and initializes the corresponding node struct to 0
 * Rewrote it slightly to use a passthrough node reference instead of
 * possibly casting Null to an art_node and then setting the type attribute
 * on the null pointer.  I dont know if this was safe, but i didn't like it. 
 * 
 * @return:  0 , in case of memory allocation error
 *           1 , if memory allocation fails  */

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
        return 1; 
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
          
// min
static inline int min(int a, int b) {
    return (a <b) ? a : b;
}

/**
 * Returns the number of prefix matches between a key and a node
 */
static int check_prefix(const art_node *node, const unsigned char *key,
                        int key_len, int depth) {
    int max_cmp = min( 
            min(node->partial_len, MAX_PREFIX_LEN),
            key_len - depth);
    int i;
    for (i=0; i < max_cmp; i++) {
       if (node->partial[i] != key[depth+i])
          return i;
    }
    return i;
}

/** Checks if leaf matches
 * @return  0 on success
 */
static int leaf_matches(const art_leaf *leaf, const unsigned char *key,
                        int key_len){
    // Fail if key length are diffrent
    if (leaf->key_len != (uint32_t)key_len) return 1;

    //Compare the keys
    return memcmp(leaf->key, key, key_len);
}

/**
 * Searches for a value in the ART tree
 * @arg t The tree
 * @arg key 
 * @arg key_len
 * @return NULL if item wasn't found, otherwise return
 *         value pointer
 */
void *art_search(const art_tree *t, const unsigned char *key, int key_len){
    art_node **children;
    art_node *node = t->root;
    int prefix_len, depth = 0;
    while (node) {
        //Check if leaf
        if (IS_LEAF(node)) {
            node = (art_node*)LEAF_RAW(node);
            // Check if the expanded path matches
            if (!leaf_matches((art_leaf*)node, key, key_len))
                return ((art_leaf*)node)->value;
            return NULL;
        }

        // Bail if prefix doesn't match 
        if (node->partial_len) {
            prefix_len = check_prefix(node, key, key_len, depth);
            if (prefix_len != min(MAX_PREFIX_LEN, node->partial_len))
                return NULL;
            depth = depth + node->partial_len;
        }

        //Recursive search
        children = find_child(node, key[depth]);
        node = (children) ? *children : NULL;
        depth++;
    }
    return NULL;
}

// Get minimum leaf under node
static art_leaf* minimum(const art_node *node) {
    // Base cases
    if (!node) return NULL;
    if (IS_LEAF(node)) return LEAF_RAW(node);
                    
    int i;
    switch (node->type) {
        case NODE4:
            return minimum(((const art_node4*)node)->children[0]);
        case NODE16:
            return minimum(((const art_node16*)node)->children[0]);
        case NODE48:
            i=0;
            while (!((const art_node48*)node)->keys[i]) i++;
            i = ((const art_node48*)node)->keys[i] -1; 
            return minimum(((const art_node48*)node)->children[i]);
        case NODE256:
            i=0;
            while (!((const art_node256*)node)->children[i]) i++;
            return minimum(((const art_node256*)node)->children[i]);
        default:
            abort();
    }
    return NULL;
}

// Get maximum leaf under node
static art_leaf* maximum(const art_node *node) {
    // Base cases
    if (!node) return NULL;
    if (IS_LEAF(node)) return LEAF_RAW(node);

    int i;
    switch (node->type) {
        case NODE4:
            return maximum(
                    ((const art_node4*)node)->children[node->num_children-1]);
        case NODE16:
            return maximum(
                    ((const art_node16*)node)->children[node->num_children-1]);
        case NODE48:
            i=255;
            while (!((const art_node48*)node)->keys[i]) i--;
            i = ((const art_node48*)node)->keys[i]-1;
            return maximum(((const art_node48*)node)->children[i]);
        case NODE256:
            i=255;
            while (!((const art_node256*)node)->children[i]) i--;
            return maximum(((const art_node256*)node)->children[i]);
        default:
            abort();
    }
    return NULL;
}

/*
 * Returns the minimum valued leaf
 */
art_leaf *art_minimum(art_tree *t) {
    return minimum((art_node*)t->root);
}

/**
 * Returns the maximum valued leaf
 */
art_leaf *art_maximum(art_tree *t) {
    return maximum((art_node*)t->root);
}

/**
 * Changed from original project
 * I use a pass through reference for the created leaf.
 * @arg nleaf reference to leaf to be filled
 * @arg *key
 * @arg key_len
 * @arg *value
 * @returns 0 if creation succeeded
 *          1 if it failed
 */
static int make_leaf(art_leaf *nleaf, const unsigned char *key,
                     int key_len, void *value) {
    nleaf = (art_leaf*)calloc(1, sizeof(art_leaf)+key_len);
    if(nleaf){
        nleaf->value = value;
        nleaf->key_len = key_len;
        memcpy(nleaf->key,key,key_len);
        return 0;
    }
    return 1;
}

/**
 * Takes two leafs and a depth as input and calculates the longest common
 * prefix;
 */
static int longest_common_prefix(art_leaf *l1, art_leaf *l2, int depth) {
    int max_cmp = min(l1->key_len, l2->key_len) - depth;
    int i;
    for (i=0; i < max_cmp; i++) {
        if (l1->key[depth+i] != l2->key[depth+i])
            return i;
    }
    return i;
}

static void copy_header(art_node *dest, art_node *src) {
    dest->num_children = src->num_children;
    dest->partial_len = src->partial_len;
    memcpy(dest->partial, src->partial, min(MAX_PREFIX_LEN, src->partial_len));
}

// Adding Children


static int add_child256(art_node256 *node, unsigned char c, void *child) {
    node->n.num_children++;
    node->children[c] = (art_node*)child;
    return 0;
}

/**
 * Adds a Node48 as a child and 
 *
 * @return: 0 if success 
 *          1 otherwise
 */
static int add_child48(art_node48 *node, art_node **ref, unsigned char c,
                        void *child) {
    int ret = 0;
    if (node->n.num_children < 48) {
        int pos = 0;
        while (node->children[pos]) pos++;
        node->children[pos] = (art_node*)child;
        node->keys[c] = pos + 1;
        node->n.num_children++;
    } else {
        /*I wanted a way to better detect allocation issues and
         * now this happend. This looks really ugly.
         * The allocated memory lives under a new pointer and
         * should be freed all the same, when the time comes.
         * In before this causes the biggest memory leak of all time */
        art_node256 *new_node;
        art_node *temp_node = NULL;
        ret = alloc_node(NODE256,temp_node);
        if(ret){
            zfree(&temp_node);
            goto endfun;
        }            
        new_node = (art_node256*)temp_node;
        zfree(&node);
        add_child256(new_node, c, child);
    }
endfun:
    return ret;
}

static int add_child16(art_node16 *node, art_node **ref,
                       unsigned char c, void *child) {
    int ret = 0;
    if (node->n.num_children < 16) {
        unsigned mask = (1 << node->n.num_children) - 1; 
        
        // non x86 architectures 
        #ifdef __i386__
            __m128i cmp;

            // Compare the key to all 16 stored keys
            cmp = _mm_cmplt_epi8( _mm_set1_epi8(c), 
                    _mm_loadu_si128((__m128i*)node->keys));

            //Use mask to ignore children that don't exist
            unsigned bitfield = _mm_movemask_epi8(cmp) & mask;
        #else
        #ifdef __amd64__
            __m128i cmp;

            //compare the key to all 16 stored keys
            cmp = _mm_cmplt_epi8( _mm_set1_epi8(c),
                    _mm_loadu_si128((__m128i*)node->keys));

            // Use mask to ignore children that don't exist
            unsigned bitfield = _mm_movemask_epi8(cmp) & mask;
        #else
            // architecture not supported, doing it the labourious way
            unsigned bitfield = 0;
            for (short i = 0; i < 16; i++) {
                if (c < node->keys[i])
                    bitfield |= (1 << i);
            }

            // Use a mask to ignore children that don't exist
            unsigned bitfield &= mask;
        #endif
        #endif

        // Check if less than any
        unsigned i;
        if (bitfield) {
             i = __builtin_ctz(bitfield);
             memmove(node->keys+i+1, node->keys+i,
                     node->n.num_children-i);
             memmove(node->children+i+1, node->children+i,
                     (node->n.num_children-i)*sizeof(void*));
        } else
             i = node->n.num_children;
        // Setting child
        node->keys[i] = c;
        node->children[i] = (art_node*)child;
        node->n.num_children++;
    } else {
        art_node48 *new_node;
        art_node *tmp_node = NULL;
        ret = alloc_node(NODE48,tmp_node);
        if(ret){
            zfree(&tmp_node);
            goto endfun;

        } 
        new_node = (art_node48*)tmp_node;
        memcpy(new_node->children, node->children,
               sizeof(void*)*node->n.num_children);
        for (int ii=0; ii<node->n.num_children; ii++) {
            new_node->keys[node->keys[ii]] = ii + 1;
        }
        copy_header((art_node*)new_node, (art_node*)node);
        *ref = (art_node*)new_node;
        zfree(&node);
        add_child48(new_node, ref, c, child);
    }
endfun: 
    return ret;
}

static int add_child4(art_node4 *node, art_node **ref,
                       unsigned char c, void *child) {
    int ret = 0;
    if (node->n.num_children < 4 ) {
        int i;
        // find child that is larger than key
        for (i=0; i < node->n.num_children; i++) {
            if (c < node->keys[i]) break;
        }

        // Shift to make room
        memmove(node->keys+i+1, node->keys+i, node->n.num_children - i);
        memmove(node->children+i+1, node->children+i,
                (node->n.num_children - i)*sizeof(void*));
        // Insert element 
        node->keys[i] = c;
        node->children[i] = (art_node*)child;
        node->n.num_children++;
        
    } else { 
        art_node16 *new_node;
        art_node *tmp_node = NULL;
        ret = alloc_node(NODE16, tmp_node);
        
        if(ret) {
            zfree(&tmp_node);
            goto endfun;
        } 
        new_node = (art_node16*)tmp_node;
        memcpy(new_node->children, node->children,
               sizeof(void*)*node->n.num_children);
        memcpy(new_node->keys, node->keys,
               sizeof(unsigned char)*node->n.num_children);
        copy_header((art_node*)new_node, (art_node*)node);
        *ref = (art_node*)new_node;
        zfree(&node);
        add_child16(new_node, ref, c, child);
        
    }
endfun:
    return ret;
}

/**
 * Wrapper function for addition of child nodes
 *
 * @return 0 if success and 1 if addition failed
 */
static int add_child(art_node *node, art_node **ref,
                     unsigned char c, void *child) {
    switch (node->type) {
        case NODE4: 
            return add_child4((art_node4*)node, ref, c, child);
        case NODE16:
            return add_child16((art_node16*)node, ref, c, child);
        case NODE48:
            return add_child48((art_node48*)node, ref, c, child);
        case NODE256:
            return add_child256((art_node256*)node, c, child);
        default:
            abort();
    }
    return 1;
}

/**
 * Calculates the index at which the prefixes mismatch
 */
static int prefix_mismatch(const art_node *node, const unsigned char *key,
                           int key_len, int depth) {
    int max_cmp = min(
                      min(MAX_PREFIX_LEN, node->partial_len),
                      key_len - depth);
    int i;
    for (i=0; i < max_cmp; i++) {
        if (node->partial[i] != key[depth+i])
            return i;
    }

    // If the prefix is short we can avoid finding a leaf
    if (node->partial_len > MAX_PREFIX_LEN) {
        // Prefix is longer than what we've checked, find leaf
        art_leaf *leaf = minimum(node);
        max_cmp = min(leaf->key_len, key_len) - depth;
        for (; i < max_cmp; i++) {
            if (leaf->key[i+depth] != key[depth+i])
                return i;
        }
    }
    return i;
}

/* The following is the interesting bit I wanted to modify 
 * 
 * I want to not automatically update existing values on insert 
 * but give the option to do something else between detection of this state 
 * and the actual update of the value. */



#define NOMINAL 0
#define ART_MEMORY_ALLOCATION_ERROR 1
#define ART_VALUE_ALLREADY_EXISTS 2

/**
 * recursive insertion of a leaf 
 *
 */

static int recursive_insert(art_node *node, art_node **ref,
                            const unsigned char *key, int key_len,
                            void *value, int depth, int *old) {
    int ret = NOMINAL;
    if (!node) {
        art_leaf *tmp_leaf = NULL;
        
        ret = make_leaf( tmp_leaf, key, key_len, value );
        if (ret == ART_MEMORY_ALLOCATION_ERROR){
            zfree(&tmp_leaf);
            goto endfun;
        }
        *ref = (art_node*)SET_LEAF(tmp_leaf);
        goto endfun;
    }

    // If we are a leaf, we need to update an existing value
    if (IS_LEAF(node)) { 
        art_leaf *leaf = LEAF_RAW(node);

        // Check if we are updating an existing value 
        if(!leaf_matches(leaf, key, key_len)) {
            printf( "I'm not creative enough to think of something clever\n");
            ret = ART_VALUE_ALLREADY_EXISTS; 
            goto endfun;
        }

        // New value, we must split the leaf into node4
        art_node4 *new_node;
        art_node *tmp_node = NULL;
        ret = alloc_node(NODE4, tmp_node);
        if ( ret == ART_MEMORY_ALLOCATION_ERROR){
            zfree(&tmp_node);
            goto endfun;
        }
        new_node = (art_node4*)tmp_node;
        
        // Create a new leaf
        art_leaf *leaf2 = NULL;
        ret = make_leaf( leaf2, key, key_len, value);
        if ( ret == ART_MEMORY_ALLOCATION_ERROR){
            goto leafInsertfreeAll;
        }
        
        //Determine longest prefix
        int longest_prefix = longest_common_prefix(leaf, leaf2, depth);
        new_node->n.partial_len = longest_prefix;
        memcpy(new_node->n.partial, key+depth,
               min(MAX_PREFIX_LEN, longest_prefix));
        
        // Add the leafs to the new node4        
        *ref = (art_node*)new_node;
        ret = add_child4(new_node, ref,
                         leaf->key[depth+longest_prefix], SET_LEAF(leaf));
        if ( ret == ART_MEMORY_ALLOCATION_ERROR)
            goto leafInsertRecoverOldState;
        //Adding leaf2
        ret = add_child4(new_node, ref,
                         leaf2->key[depth+longest_prefix], SET_LEAF(leaf2));
        if ( ret == ART_MEMORY_ALLOCATION_ERROR)
           goto leafInsertRecoverOldState; 

        //Cleanup in case of error 
        if(0){
            //Restore old state
leafInsertRecoverOldState:
            /* I needed a lot of time to notice that freeing a node 
             * doesn't automatically free its child nodes.
             * Freeing new_node shouldn't touch the original leaf memory*/
            *ref = (art_node*)SET_LEAF(leaf); 
leafInsertfreeAll:
            zfree(&leaf2);
            zfree(&new_node);
            goto endfun;
        }
    }
    // Case Node: Check if given Node has a prefix
    if (node->partial_len) {
        // Determine if the prefixes differ, since we need to split
        int prefix_diff = prefix_mismatch(node, key, key_len,depth);
        if ((uint32_t)prefix_diff >= node->partial_len) {
            depth += node->partial_len;
            goto RECURSE_SEARCH;
        }
        /* 
         * This char will later join the ranks of child nodes
         * and might be hard to recover, in case we have to undo this
         * operation quickly. From here on out, it will be known as 
         * the safety char of justice. 
         */      
        char recovery_key = node->partial[prefix_diff];
        int keyIsToBig = 0;
        //Create a new node
        art_node *tmp_node = NULL;
        ret = alloc_node(NODE4, tmp_node);
        if (ret == ART_MEMORY_ALLOCATION_ERROR){
            zfree(&tmp_node);
            goto endfun;
        }
        art_node4 *new_node = (art_node4*)tmp_node;
        /* 
         * after this line recovery from fault becomes tricky 
         * We are linking the new_node to the place our old node 
         * occupied
         */
        *ref = (art_node*)new_node;
        new_node->n.partial_len = prefix_diff;
        memcpy(new_node->n.partial, node->partial,
               min(MAX_PREFIX_LEN, prefix_diff));

        // Adjust the prefix of the old node
        if (node->partial_len <= MAX_PREFIX_LEN) {
            // Adding old node to new node
            ret = add_child4(new_node, ref, node->partial[prefix_diff], node);  
            if (ret == ART_MEMORY_ALLOCATION_ERROR)
                goto nodeInsertRecoverNewNode;
            node->partial_len -= (prefix_diff+1);            
            memmove(node->partial,node->partial+prefix_diff+1,
                    min(MAX_PREFIX_LEN, node->partial_len));
        } else {
            keyIsToBig = 1;
            node->partial_len -= (prefix_diff+1);
            /* 
             * We cannot get the required key information from the partial,
             * therefore we need to retrieve it from the smallest leaf
             * below us.
             */
            art_leaf *leaf = minimum(node);
            ret = add_child4(new_node, ref, leaf->key[depth+prefix_diff], node);              
            if ( ret == ART_MEMORY_ALLOCATION_ERROR)
                goto nodeInsertRecoverNewNode;
            /*
             * I believe the min to be redundant at this point, because the
             * partial length is larger than MAX_PREFIX_LENGTH or else we
             * shouldn't be in this condition.
             */
            memcpy(node->partial, leaf->key+depth+prefix_diff+1,
                   min(MAX_PREFIX_LEN, node->partial_len));
        }

        // Insert the new leaf
        art_leaf *new_leaf = NULL;
        ret = make_leaf( new_leaf, key, key_len, value);
        if ( ret == ART_MEMORY_ALLOCATION_ERROR)  
            goto nodeInsertRecoverAll;
        ret = add_child4(new_node, ref, 
                         key[depth+prefix_diff], SET_LEAF(new_leaf));
        goto endfun;

nodeInsertRecoverAll:
        if (!keyIsToBig){
            // Recover from memmove in node
            memmove(node->partial+prefix_diff+1,node->partial,
                    min(MAX_PREFIX_LEN, node->partial_len));
            node->partial[prefix_diff] = recovery_key; 
        }   
        /* 
         * This index pushing is likely to be wrong
         */
        memcpy(node->partial, new_node->n.partial, 
               min(MAX_PREFIX_LEN, prefix_diff));

nodeInsertRecoverNewNode:
            zfree(&new_node);
            *ref = node;
            goto endfun;            
    }
        
RECURSE_SEARCH:;
    // Find a child to recurse to
    art_node **child = find_child(node, key[depth]);
    if (child) { 
        return recursive_insert(*child, child, key, key_len,
                                value, depth+1 ,old);
    } 

    // No child, node goes within us
    art_leaf *leaf = NULL;
    ret = make_leaf(leaf, key, key_len, value);
    if (ret == ART_MEMORY_ALLOCATION_ERROR){
        goto freeRecurseLeaf;
    }
    ret = add_child(node, ref, key[depth], SET_LEAF(leaf));
    if (ret == ART_MEMORY_ALLOCATION_ERROR)
        goto freeRecurseLeaf;
        
endfun:
    return ret;

freeRecurseLeaf:
    zfree(&leaf);
    return ret;
}

/**
 * Inserts a new value into the ART tree 
 * @arg tree The tree
 * @arg key The key
 * @arg key_len
 * @arg value Opaque value
 * @return int 
            0 - NOMINAL
            1 - ART_MEMORY_ALLOCATION_ERROR
            2 - ART_VALUE_ALLREADY_EXISTS
            
 */

int art_insert(art_tree *tree, const unsigned char *key,
               int key_len, void *value) {
    int ret = NOMINAL;
    int old_val = 0;
    int *depth = 0;
    ret = recursive_insert(tree->root, &tree->root, key,
                           key_len, value, 0, &old_val);
    switch (ret){     
        case NOMINAL:
            break;
        case ART_VALUE_ALLREADY_EXISTS:
            break;
        case ART_MEMORY_ALLOCATION_ERROR:
            break;
        default:
            break; 
    }
    return ret;
}

static int remove_child256(art_node256 *node, 
                            art_node **ref, unsigned char c ) {
    int ret = NOMINAL;
    node->children[c] = NULL;
    node->n.num_children--;
    
    /*
     * Resize to node48 on underflow, not immedialtly to prevent
     * thrashing if we sit on the 48/49 boundary
     */
    if (node->n.num_children == 37) {
        art_node *tmp_node = NULL;
        ret = alloc_node(NODE48, tmp_node);
        if (ret == ART_MEMORY_ALLOCATION_ERROR){
            zfree(&tmp_node);
            goto endfun;
        }
        // Could use tmp_node, but that feels wrong  
        art_node48 *new_node = (art_node48*)tmp_node;
        *ref = (art_node*)new_node;
        copy_header((art_node*)new_node, (art_node*)node);

        int pos = 0;
        for (int i=0; i<256;i++) { 
            if (node->children[i]){
                new_node->children[pos] = node->children[i];
                new_node->keys[i] = pos + 1;
                pos++;
            }
        }
        zfree(&node);
    }
endfun:
    return ret;
}

static int remove_child48(art_node48 *node, art_node **ref, unsigned char c) {
    int ret = NOMINAL;
    int pos = node->keys[c];
    node->keys[c] = 0;
    node->children[pos-1] = NULL;
    node->n.num_children--;

    if (node->n.num_children == 12) {
        art_node *tmp_node = NULL;
        ret = alloc_node(NODE16, tmp_node);
        if (ret == ART_MEMORY_ALLOCATION_ERROR){
            zfree(&tmp_node);
            goto endfun;
        }
        art_node16 *new_node = (art_node16*)tmp_node;
        *ref = (art_node*)new_node;
        copy_header((art_node*)new_node, (art_node*)node);

        int child = 0;
        for( int i=0; i < 256; i++) {
            pos = node ->keys[i];
            if (pos) {
                new_node->keys[child] = i;
                new_node->children[child] = node->children[pos-1];
                child++;
            }
        }
        zfree(&node);
    }        
endfun:
    return ret;
}


