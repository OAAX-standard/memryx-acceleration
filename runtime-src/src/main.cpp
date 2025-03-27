#include "runtime_core.hpp"
#include <stdio.h>
#include <stdlib.h>

char *read_json(const char *json_path){
    FILE *fp;
    long lSize;
    char *buffer;

    fp = fopen(json_path, "rb");
    if (!fp){
        printf("Error: cannot open the JSON file\n");
        exit(1);
        return NULL;
    }

    fseek(fp, 0L, SEEK_END);
    lSize = ftell(fp);
    rewind(fp);

    buffer = (char *)malloc(lSize + 1);
    if (!buffer){
        printf("Error: cannot allocate memory\n");
        fclose(fp);
        return NULL;
    }

    if (1 != fread(buffer, lSize, 1, fp)){
        fclose(fp);
        free(buffer);
        return NULL;
    }

    fclose(fp);
    return buffer;
}

int main(int argc, char *argv[]){
    if (argc < 2){
        printf("Usage: %s <model_path>\n", argv[0]);
        return 1;
    }
    char *model_path = argv[1];
    const char *json_path = "io.json";
    char *json = read_json(json_path);
    const char *keys[] = {"json"};
    const void *values[] = {json};

    int exit_code = runtime_initialization_with_args(1, keys, values);
    printf("runtime_initialization_with_args's exit_code: %d\n", exit_code);
    
    exit_code = runtime_model_loading(model_path);
    printf("runtime_model_loading's exit_code: %d\n", exit_code);


    return 0;
}