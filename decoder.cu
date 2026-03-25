#include <cstdio>
#include <cstdlib>
#include <string>

struct Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};

int load_config(Config& config, std::string& model_file) {
    FILE* f = fopen(model_file.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "failed to open: %s\n", model_file.c_str());
        return -1;
    }

    if (fread(&config, sizeof(Config), 1, f) != 1) {
        fprintf(stderr, "failed to read config\n");
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_file>\n", argv[0]);
        return 1;
    }

    Config config;
    std::string model_file = argv[1];
    if (load_config(config, model_file) != 0) {
        return 1;
    }

    printf("dim        = %d\n", config.dim);
    printf("hidden_dim = %d\n", config.hidden_dim);
    printf("n_layers   = %d\n", config.n_layers);
    printf("n_heads    = %d\n", config.n_heads);
    printf("n_kv_heads = %d\n", config.n_kv_heads);
    printf("vocab_size = %d\n", config.vocab_size);
    printf("seq_len    = %d\n", config.seq_len);

    return 0;
}