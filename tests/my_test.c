
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include "../src/art.h"



int fail_unless(int test_result) { return test_result;}

int pure_init(){
    int ret ;
    art_tree t;
    int res = art_tree_init(&t);
    ret = fail_unless(res == 0);

    ret = fail_unless(art_size(&t) == 0);

    res = art_tree_destroy(&t);
    ret = fail_unless(res == 0);
    return ret; 
}

int insert_init_test(){
    int ret = 0;
    art_tree t;
    int res = art_tree_init(&t);
    ret = fail_unless(res == 0);

    int len;
    char buf[512];
    FILE *f = fopen("words.txt", "r");

    uintptr_t line = 1;
    while (fgets(buf, sizeof buf, f)) {
        len = strlen(buf);
        buf[len-1] = '\0';
        ret = fail_unless(0 == art_insert(&t, (unsigned char*)buf, len, (void*)line));
        ret = fail_unless(art_size(&t) == line);
        line++;
    }

    res = art_tree_destroy(&t);
    ret = fail_unless(res == 0);

    return ret;
}

int test_art_insert_verylong(){
    int ret = 0;
    art_tree t;
    int res = art_tree_init(&t);
    fail_unless(res == 0);

    unsigned char key1[300] = {16,0,0,0,7,10,0,0,0,2,17,10,0,0,0,120,10,0,0,0,120,10,0,
      0,0,216,10,0,0,0,202,10,0,0,0,194,10,0,0,0,224,10,0,0,0,
      230,10,0,0,0,210,10,0,0,0,206,10,0,0,0,208,10,0,0,0,232,
      10,0,0,0,124,10,0,0,0,124,2,16,0,0,0,2,12,185,89,44,213,
      251,173,202,211,95,185,89,110,118,251,173,202,199,101,0,
      8,18,182,92,236,147,171,101,150,195,112,185,218,108,246,
      139,164,234,195,58,177,0,8,16,0,0,0,2,12,185,89,44,213,
      251,173,202,211,95,185,89,110,118,251,173,202,199,101,0,
      8,18,180,93,46,151,9,212,190,95,102,178,217,44,178,235,
      29,190,218,8,16,0,0,0,2,12,185,89,44,213,251,173,202,
      211,95,185,89,110,118,251,173,202,199,101,0,8,18,180,93,
      46,151,9,212,190,95,102,183,219,229,214,59,125,182,71,
      108,180,220,238,150,91,117,150,201,84,183,128,8,16,0,0,
      0,2,12,185,89,44,213,251,173,202,211,95,185,89,110,118,
      251,173,202,199,101,0,8,18,180,93,46,151,9,212,190,95,
      108,176,217,47,50,219,61,134,207,97,151,88,237,246,208,
      8,18,255,255,255,219,191,198,134,5,223,212,72,44,208,
      250,180,14,1,0,0,8, '\0'};
    unsigned char key2[303] = {16,0,0,0,7,10,0,0,0,2,17,10,0,0,0,120,10,0,0,0,120,10,0,
      0,0,216,10,0,0,0,202,10,0,0,0,194,10,0,0,0,224,10,0,0,0,
      230,10,0,0,0,210,10,0,0,0,206,10,0,0,0,208,10,0,0,0,232,
      10,0,0,0,124,10,0,0,0,124,2,16,0,0,0,2,12,185,89,44,213,
      251,173,202,211,95,185,89,110,118,251,173,202,199,101,0,
      8,18,182,92,236,147,171,101,150,195,112,185,218,108,246,
      139,164,234,195,58,177,0,8,16,0,0,0,2,12,185,89,44,213,
      251,173,202,211,95,185,89,110,118,251,173,202,199,101,0,
      8,18,180,93,46,151,9,212,190,95,102,178,217,44,178,235,
      29,190,218,8,16,0,0,0,2,12,185,89,44,213,251,173,202,
      211,95,185,89,110,118,251,173,202,199,101,0,8,18,180,93,
      46,151,9,212,190,95,102,183,219,229,214,59,125,182,71,
      108,180,220,238,150,91,117,150,201,84,183,128,8,16,0,0,
      0,3,12,185,89,44,213,251,133,178,195,105,183,87,237,150,
      155,165,150,229,97,182,0,8,18,161,91,239,50,10,61,150,
      223,114,179,217,64,8,12,186,219,172,150,91,53,166,221,
      101,178,0,8,18,255,255,255,219,191,198,134,5,208,212,72,
      44,208,250,180,14,1,0,0,8, '\0'};


    fail_unless(0 == art_insert(&t, key1, 299, (void*)key1));
    fail_unless(0 == art_insert(&t, key2, 302, (void*)key2));
    art_insert(&t, key2, 302, (void*)key2);
    fail_unless(art_size(&t) == 2);

    res = art_tree_destroy(&t);
    ret = fail_unless(res == 0);
    return ret;
}

int test_art_insert_search(){
    int ret = 0;
    art_tree t;
    int res = art_tree_init(&t);
    fail_unless(res == 0);

    int len;
    char buf[512];
    FILE *f = fopen("words.txt", "r");

    uintptr_t line = 1;
    while (fgets(buf, sizeof buf, f)) {
        len = strlen(buf);
        buf[len-1] = '\0';
        fail_unless(0 ==
            art_insert(&t, (unsigned char*)buf, len, (void*)line));
        line++;
    }

    // Seek back to the start
    fseek(f, 0, SEEK_SET);

    // Search for each line
    line = 1;
    while (fgets(buf, sizeof buf, f)) {
        len = strlen(buf);
        buf[len-1] = '\0';

        uintptr_t val = (uintptr_t)art_search(&t, (unsigned char*)buf, len);
	    ret = fail_unless(line == val);
        fprintf(stderr,  "Line: %lu Val: %" PRIuPTR " Str: %s\n",
                   line, val, buf);
        //    return 1;
        line++;
    }

    // Check the minimum
    art_leaf *l = art_minimum(&t);
    fail_unless(l && strcmp((char*)l->key, "A") == 0);

    // Check the maximum
    l = art_maximum(&t);
    fail_unless(l && strcmp((char*)l->key, "zythum") == 0);

    res = art_tree_destroy(&t);
    fail_unless(res == 0);
    return ret;
}



int main (void){
    //pure_init();
    //insert_init_test();
    //test_art_insert_verylong();
    test_art_insert_search();
    return 0;
} 
