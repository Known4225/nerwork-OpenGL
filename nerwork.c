/* nerwork in c and openGL */

#include "include/turtle.h"
#include <time.h>

typedef struct {
    list_t *nodes; // list of nodes (2D array)
    list_t *weights; // list of weights (3D array)
    list_t *weightedSums; // list of weighted sums (2D array, just node values reversed through activation function with the first layer omitted)
    list_t *gradient; // gradient matrix (2D array)
    list_t *biases; // list of biases (2D array)
    list_t *format; // list for formatting drawing
    double wireThresh; // variable for culling wires with low (or high) values (value from 0.5 to 1 or -0.5 to -1)
    char doWireCulling; // variable for toggling impact of wireThresh on the actual nerwork output, just for fun (0 for false, 1 for true, make sure it's set to false if you want to train)

    char debug; // unused

    int sample; // current loaded sample
    list_t *data; // training data [2D array]
    list_t *pres; // list loaded in with a sample determining the expected result of that sample
    
    const char *trainingFileName;
    const char *saveFileName;
    FILE *trainingFile; // training data file
    FILE *saveFile; // save data file

    int layers; // layers of the network
    list_t *nodesPerLayer; // list of nodes per layer
} class;
void nerworkInit(class *selfp) { // initialise values
    class self = *selfp;
    self.nodes = list_init();
    self.weights = list_init();
    self.weightedSums = list_init();
    self.gradient = list_init();
    self.biases = list_init();
    self.format = list_init();
    self.wireThresh = 0;
    self.doWireCulling = 0;

    self.debug = 0;

    self.sample = 0;
    self.data = list_init();
    self.pres = list_init();
    self.trainingFileName = "null";
    self.saveFileName = "null";
    self.trainingFile = NULL;
    self.saveFile = NULL;

    self.layers = 4; // default value
    self.nodesPerLayer = list_init();
    
    *selfp = self;
}

#define ACTIVATION_FUNCTION(inp) (1 / (1 + exp(-inp))) // activation function macro (sigmoid)
#define DERIV_ACTIVATION_FUNCTION(inp) (ACTIVATION_FUNCTION(inp) * (1 - ACTIVATION_FUNCTION(inp))) // derivative of activation function (specific sigmoid derivative)

void process(class *selfp) { // calculates neural network result given layer 1 nodes have been set
    class self = *selfp;
    for (int i = 1; i < self.layers; i++) {
        for (int j = 0; j < ((list_t*) (self.nodes -> data[i].p)) -> length; j++) {
            double acc = 0;
            for (int k = 0; k < ((list_t*) (self.nodes -> data[i - 1].p)) -> length; k++) {
                acc += (((list_t*) (((list_t*) (self.weights -> data[i - 1].p)) -> data[k].p)) -> data[j].d) * (((list_t*) (self.nodes -> data[i - 1].p)) -> data[k].d); // calculate weighted sum
            }
            acc += ((list_t*) (self.biases -> data[i].p)) -> data[j].d;
            ((list_t*) (self.weightedSums -> data[i].p)) -> data[j] = (unitype) acc;
            ((list_t*) (self.nodes -> data[i].p)) -> data[j] = (unitype) ACTIVATION_FUNCTION(acc);
        }
    }
    *selfp = self;
}
int loadTrainingInstance(class *selfp, int instance) { // loads an instance of the training data to the network
    class self = *selfp;
    if (instance >= self.data -> length) {
        printf("No Sample %d\n", instance);
        return -1;
    }
    if (((list_t*) (self.data -> data[instance].p)) -> length - 1 > ((list_t*) (self.nodes -> data[0].p)) -> length) {
        printf("Error: not enough layer 1 nodes\n");
        return -1;
    }
    for (int i = 0; i < ((list_t*) (self.data -> data[instance].p)) -> length; i++) {
        ((list_t*) (self.nodes -> data[0].p)) -> data[((i - 1) % 28) * 28 + ((i - 1) / 28)] = (unitype) (((double) ((list_t*) (self.data -> data[instance].p)) -> data[i].i) / 255);
        /*
        because my renderer goes like
        1  5  9  13
        2  6  10 14
        3  7  11 15
        4  8  12 16

        and the data is like
        1  2  3  4
        5  6  7  8
        9 10 11 12
        13 14 15 16

        some translation calculations must take place
        specifically, switching mod and divison
        */
    }
    process(&self);
    self.sample = instance;
    list_clear(self.pres);
    for (int i = 0; i < ((list_t*) (self.nodes -> data[self.layers - 1].p)) -> length; i++) {
        list_append(self.pres, (unitype) (double) 0, 'd');
    }
    self.pres -> data[(int) ((list_t*) (self.data -> data[self.sample].p)) -> data[0].i] = (unitype) (double) 1;
    *selfp = self;
    return 0;
}
int setup(class *selfp) { // setup the network using self.layers and self.nodesPerLayer, randomise weights and biases
    class self = *selfp;
    list_clear(self.nodes);
    list_clear(self.weights);
    list_clear(self.weightedSums);
    list_clear(self.gradient);
    list_clear(self.biases);
    if (self.layers != self.nodesPerLayer -> length) {
        printf("setup layer mismatch!\n");
        return -1; // error return
    }
    for (int i = 0; i < self.layers; i++) {
        list_append(self.nodes, (unitype) (void*) list_init(), 'r');
        list_append(self.weightedSums, (unitype) (void*) list_init(), 'r');
        list_append(self.gradient, (unitype) (void*) list_init(), 'r');
        list_append(self.weights, (unitype) (void*) list_init(), 'r');
        list_append(self.biases, (unitype) (void*) list_init(), 'r');
        for (int j = 0; j < self.nodesPerLayer -> data[i].i; j++) {
            list_append((list_t*) (self.weights -> data[i].p), (unitype) (void*) list_init(), 'r');
            list_append((list_t*) (self.nodes -> data[i].p), (unitype) (double) 0, 'd');
            list_append((list_t*) (self.weightedSums -> data[i].p), (unitype) (double) 0, 'd');
            list_append((list_t*) (self.gradient -> data[i].p), (unitype) (double) 0, 'd');
            if (i > 0) {
                list_append((list_t*) (self.biases -> data[i].p), (unitype) (((double) rand() / RAND_MAX - 0.5) * 2), 'd');
            }
            if (i < self.layers - 1) {
                for (int k = 0; k < self.nodesPerLayer -> data[i + 1].i; k++) {
                    list_append((list_t*) (self.gradient -> data[i].p), (unitype) (double) 0, 'd');
                    list_append((list_t*) (((list_t*) (self.weights -> data[i].p)) -> data[j].p), (unitype) (((double) rand() / RAND_MAX - 0.5) * 2), 'd');
                }
            }
        }
    }
    list_pop(self.gradient);
    list_pop(self.weights);
    *selfp = self;
    return 0;
}
void randomiseWeightsAndBiases(class *selfp) {
    class self = *selfp;
    for (int i = 1; i < self.layers; i++) {
        for (int j = 0; j < ((list_t*) (self.nodes -> data[i].p)) -> length; j++) {
            for (int k = 0; k < ((list_t*) (self.nodes -> data[i - 1].p)) -> length; k++) {
                ((list_t*) (((list_t*) (self.weights -> data[i - 1].p)) -> data[k].p)) -> data[j] = (unitype) (((double) rand() / RAND_MAX - 0.5) * 2);
            }
            ((list_t*) (self.biases -> data[i].p)) -> data[j] = (unitype) (((double) rand() / RAND_MAX - 0.5) * 2);
        }
    }
    *selfp = self;
}
void clearInp(class *selfp) { // sets all first layer nodes to 0
    class self = *selfp;
    for (int i = 0; i < ((list_t*) (self.nodes -> data[0].p)) -> length; i++) {
        ((list_t*) (self.nodes -> data[0].p)) -> data[i] = (unitype) (double) 0;
    }
    *selfp = self;
}

/* training functions */

void adjustWeightsAndBiases(class *selfp, double scale) { // adjust weights and biases according to self.gradient
    class self = *selfp;
    for (int i = 1; i < self.layers; i++) {
        for (int j = 0; j < ((list_t*) (self.nodes -> data[i].p)) -> length; j++) {
            for (int k = 0; k < ((list_t*) (self.nodes -> data[i - 1].p)) -> length; k++) {
                double winit = ((list_t*) (((list_t*) (self.weights -> data[i - 1].p)) -> data[k].p)) -> data[j].d;
                ((list_t*) (((list_t*) (self.weights -> data[i - 1].p)) -> data[k].p)) -> data[j] = (unitype) (winit - (scale * ((list_t*) (self.gradient -> data[i - 1].p)) -> data[j * (((list_t*) (self.nodes -> data[i - 1].p)) -> length + 1) + k].d));
            }
            double binit = ((list_t*) (self.biases -> data[i].p)) -> data[j].d;
            ((list_t*) (self.biases -> data[i].p)) -> data[j] = (unitype) (binit - (scale * ((list_t*) (self.gradient -> data[i - 1].p)) -> data[(j + 1) * (((list_t*) (self.nodes -> data[i - 1].p)) -> length + 1) - 1].d));
            // printf("double: %.17lf, ", (scale * ((double) 1 / self.data -> length) * ((list_t*) (self.gradient -> data[i - 1].p)) -> data[(j + 1) * (((list_t*) (self.nodes -> data[i - 1].p)) -> length + 1) - 1].d));
            // printf("index: %d, ", (j + 1) * (((list_t*) (self.nodes -> data[i - 1].p)) -> length + 1) - 1);
            // printf("gradient: %.17lf\n", ((list_t*) (self.gradient -> data[i - 1].p)) -> data[(j + 1) * (((list_t*) (self.nodes -> data[i - 1].p)) -> length + 1) - 1].d);
        }
    }
    *selfp = self;
}
double calculateCost(class *selfp) { // calculates cost of current loaded data relative to presumed correct response (self.pres)
    class self = *selfp;
    if ((((list_t*) (self.nodes -> data[self.layers - 1].p)) -> length) != (self.pres -> length)) {
        return -1;
    }
    double acc = 0;
    for (int i = 0; i < (((list_t*) (self.nodes -> data[self.layers - 1].p)) -> length); i++) {
        acc += ((((list_t*) (self.nodes -> data[self.layers - 1].p)) -> data[i].d - self.pres -> data[i].d) * (((list_t*) (self.nodes -> data[self.layers - 1].p)) -> data[i].d - self.pres -> data[i].d));
    }
    *selfp = self;
    return acc;
}
double calculateTotalCost(class *selfp) {
    class self = *selfp;
    double acc = 0;
    for (int i = 0; i < self.data -> length; i++) {
        loadTrainingInstance(&self, i);
        acc += calculateCost(&self);
    }
    *selfp = self;
    return acc;
}
void backProp(class *selfp) { // one iteration of backpropgation, sets the self.gradient list
    class self = *selfp;
    list_t *lastLayer = list_init(); // 1D list containing all of the derivatives of a particular layer (the last refers to the last computation which moves from the last layer in the network to the first as we backpropagate)
    for (int i = 0; i < ((list_t*) (self.nodes -> data[self.layers - 1].p)) -> length; i++) { // load derivatives of the output layer
        list_append(lastLayer, (unitype) (2 * (((list_t*) (self.nodes -> data[self.layers - 1].p)) -> data[i].d - self.pres -> data[i].d)), 'd');
    }
    for (int i = self.layers - 1; i > 0; i--) { // do layers - 1 cycles, starting at layers - 1 and ending at 1
        int lengthNode = ((list_t*) (self.nodes -> data[i - 1].p)) -> length;
        for (int j = 0; j < ((list_t*) (self.nodes -> data[i].p)) -> length; j++) {
            double derivActivate = DERIV_ACTIVATION_FUNCTION(((list_t*) (self.weightedSums -> data[i].p)) -> data[j].d);
            double lastLayerJ = lastLayer -> data[j].d;
            for (int k = 0; k < lengthNode; k++) {
                ((list_t*) (self.gradient -> data[i - 1].p)) -> data[j * (lengthNode + 1) + k] = (unitype) (((list_t*) (self.nodes -> data[i - 1].p)) -> data[k].d * derivActivate * lastLayerJ); // set weight
            }
            ((list_t*) (self.gradient -> data[i - 1].p)) -> data[(j + 1) * (lengthNode + 1) - 1] = (unitype) (1 * derivActivate * lastLayerJ); // set bias
        
            // gradient and weight lists have lengths of layers - 1, reflecting how there are more layers of nodes than layers of weights. The bias list index matches that of the nodes, but index 0 of the bias list is empty
            // self.weight[i - 1][k][j] represents self.gradient[i - 1][j * (len(self.nodes[i - 1]) + 1) + k], which is a connection from self.nodes[i - 1][k] to self.nodes[i][j]

        }
        if (i != 1) {
            list_t *lastLayer2 = list_init();
            list_copy(lastLayer, lastLayer2); // create copy of lastLayer to change weights
            list_clear(lastLayer); // setup lastLayer for next backprop iteration (only happens layers - 2 times)
            for (int j = 0; j < lengthNode; j++) {
                double acc = 0;
                for (int k = 0; k < ((list_t*) (self.nodes -> data[i].p)) -> length; k++) {
                    acc += ((list_t*) (((list_t*) (self.weights -> data[i - 1].p)) -> data[j].p)) -> data[k].d * DERIV_ACTIVATION_FUNCTION(((list_t*) (self.weightedSums -> data[i].p)) -> data[k].d) * lastLayer2 -> data[k].d;
                }
                list_append(lastLayer, (unitype) acc, 'd');
            }
            list_free(lastLayer2);
        }
    }
    list_free(lastLayer);
    *selfp = self;
}

/* render function(s) */

void transform(class *selfp, int factor) {
    class self = *selfp;
    int len = ((list_t*) (self.nodes -> data[0].p)) -> length;
    if (factor > 0) {
        for (int i = 0; i < len; i++) { // go forward
            if (i + factor < len) {
                ((list_t*) (self.nodes -> data[0].p)) -> data[i] = ((list_t*) (self.nodes -> data[0].p)) -> data[i + factor];
            } else {
                ((list_t*) (self.nodes -> data[0].p)) -> data[i] = ((list_t*) (self.nodes -> data[0].p)) -> data[i + factor - len];
            }
        }
    } else {
        for (int i = len - 1; i > -1; i--) { // go reverse
            if (i + factor > -1) {
                ((list_t*) (self.nodes -> data[0].p)) -> data[i] = ((list_t*) (self.nodes -> data[0].p)) -> data[i + factor];
            } else {
                ((list_t*) (self.nodes -> data[0].p)) -> data[i] = ((list_t*) (self.nodes -> data[0].p)) -> data[i + factor + len];
            }
        }
    }
    *selfp = self;
}
void drawNetwork(class *selfp, char nodeValues, char wires) { // renders the network
    class self = *selfp;
    turtleClear();
    list_t *initPositions = list_init();
    double size = 22;
    if (self.format -> length < 1) {
        list_append(self.format, (unitype) "Left", 's');
    }
    double totalXlen = 0;
    for (int i = 0; i < self.layers; i++) {
        if (self.format -> length < (i + 2)) {
            list_append(self.format, (unitype) (int) (((list_t*) (self.nodes -> data[i].p)) -> length), 'i');
        }
        totalXlen += ((((list_t*) (self.nodes -> data[i].p)) -> length) / self.format -> data[i + 1].i) * size * 1.1;
        totalXlen += size * 2;
    }
    double x = -totalXlen / 2 * 0.5;
    double y;
    double maxY = 0;
    for (int i = 0; i < self.layers; i++) {
        list_append(initPositions, (unitype) x, 'd');
        list_append(initPositions, (unitype) (size * 0.275 * (self.format -> data[i + 1].i - 1)), 'd');
        x += (size * 0.55 * (((((list_t*) (self.nodes -> data[i].p)) -> length) / self.format -> data[i + 1].i) - 1)) + size * 2;
        if (initPositions -> data[initPositions -> length - 1].d > maxY) {
            maxY = initPositions -> data[initPositions -> length - 1].d;
        }
    }
    if (wires) {
        turtlePenSize(1);
        for (int i = 0; i < self.layers - 1; i++) {
            for (int j = 0; j < ((list_t*) (self.nodes -> data[i].p)) -> length; j++) {
                for (int k = 0; k < ((list_t*) (((list_t*) (self.weights -> data[i].p)) -> data[j].p)) -> length; k++) {
                    // double val = (((list_t*) (((list_t*) (self.weights -> data[i].p)) -> data[j].p)) -> data[k].d);
                    double sig = (1 / (1 + exp(fabs(((list_t*) (((list_t*) (self.weights -> data[i].p)) -> data[j].p)) -> data[k].d))));
                    int col = (int) (255 - abs(255 - round(fmod(sig * 255, 255)) - 127.5) * 2); // this is good programming
                    if ((1 - sig > self.wireThresh && self.wireThresh > 0) || (1 - sig < -self.wireThresh && self.wireThresh < 0)) {
                        if (self.wireThresh > 0) {
                            turtlePenColor(col, col, col); // uses shifted sigmoid for weight 'weights' (how dark the connection appears when drawn as a function of how large its absolute value is)
                        } else {
                            turtlePenColor(255 - col, 255 - col, 255 - col);
                        }
                        turtleGoto(initPositions -> data[i * 2].d + size * 0.55 * (j / (self.format -> data[i + 1].i)), initPositions -> data[i * 2 + 1].d - size * 0.55 * (j % (self.format -> data[i + 1].i)));
                        turtlePenDown();
                        turtleGoto(initPositions -> data[i * 2 + 2].d + size * 0.55 * (k / (self.format -> data[i + 2].i)), initPositions -> data[i * 2 + 3].d - size * 0.55 * (k % (self.format -> data[i + 2].i)));
                        turtlePenUp();
                    }
                    // if (sig > 0.49)
                    //     printf("val: %lf, sig: %lf, col: %d\n", val, sig, col);
                }
            }
        }
    }
    for (int i = 0; i < self.layers; i++) {
        for (int j = 0; j < ((list_t*) (self.nodes -> data[i].p)) -> length; j++) {
            x = initPositions -> data[i * 2].d + size * 0.55 * (j / (self.format -> data[i + 1].i));
            y = initPositions -> data[i * 2 + 1].d - size * 0.55 * (j % (self.format -> data[i + 1].i));
            turtleGoto(x, y);
            turtlePenSize(size);
            turtlePenColor(0, 0, 0);
            turtlePenDown();
            turtlePenUp();
            turtlePenSize(size * 0.8);
            int col = round(255 * ((list_t*) (self.nodes -> data[i].p)) -> data[j].d);
            turtlePenColor(col, col, col);
            turtlePenDown();
            turtlePenUp();
        }
    }
    list_free(initPositions);
    *selfp = self;
}

/* file functions */

int saveWeightsAndBiases(class *selfp, const char *filename) {
    class self = *selfp;
    FILE *newFile = fopen(filename, "w");
    fprintf(newFile, "%d ", self.layers);
    for (int i = 0; i < self.layers; i++) {
        fprintf(newFile, "%d ", self.nodesPerLayer -> data[i].i);
    }
    fprintf(newFile, "\nWeights: \n");
    for (int i = 1; i < self.layers; i++) {
        for (int j = 0; j < ((list_t*) (self.nodes -> data[i].p)) -> length; j++) {
            for (int k = 0; k < ((list_t*) (self.nodes -> data[i - 1].p)) -> length; k++) {
                char toWrite[50];
                sprintf(toWrite, "%lf", ((list_t*) (((list_t*) (self.weights -> data[i - 1].p)) -> data[k].p)) -> data[j].d);
                fprintf(newFile, "%s ", toWrite);
            }
            fprintf(newFile, "\n");
        }
        fprintf(newFile, "\n");
    }
    fprintf(newFile, "Biases: \n");
    for (int i = 1; i < self.layers; i++) {
        for (int j = 0; j < ((list_t*) (self.nodes -> data[i].p)) -> length; j++) {
            char toWrite[50];
            sprintf(toWrite, "%lf", ((list_t*) (self.biases -> data[i].p)) -> data[j].d);
            fprintf(newFile, "%s ", toWrite);
        }
        fprintf(newFile, "\n");
    }
    fclose(newFile);
    *selfp = self;
    return 0;
}
int saveDataset(class *selfp, const char *filename) {
    class self = *selfp;
    FILE *newFile = fopen(filename, "a");
    fseek(newFile, 0, SEEK_END); // check size of file (bytes)
    unsigned int fileSize = ftell(newFile);
    if (fileSize < 1) { // if file is empty
        printf("creating file %s\n", filename);
        if (strcmp(self.trainingFileName, "null") != 0) { // copy existing dataset into new file
            self.trainingFile = fopen(self.trainingFileName, "rb");
            fseek(self.trainingFile, 0, SEEK_END);
            fileSize = ftell(self.trainingFile);
            printf("copying %d bytes from %s\n", fileSize, self.trainingFileName);
            fseek(self.trainingFile, 0, SEEK_SET);
            for (int i = 0; i < fileSize; i++) {
                char charWrite;
                fread(&charWrite, 1, 1, self.trainingFile);
                fwrite(&charWrite, 1, 1, newFile);
            }
            fseek(newFile, 0, SEEK_END); // ensure pointer is at the end of the file (for append)
            fclose(self.trainingFile);
        }
    }
    int toWrite;
    printf("Enter digit: ");
    fflush(stdin);
    scanf("%1d", &toWrite);
    fprintf(newFile, "%d,", toWrite);
    list_append(self.data, (unitype) (void*) list_init(), 'r');
    list_append((list_t*) (self.data -> data[self.data -> length - 1].p), (unitype) toWrite, 'i');
    for (int i = 0; i < ((list_t*) (self.nodes -> data[0].p)) -> length; i++) { // write node layer 1 data to file
        toWrite = (int) round(((list_t*) (self.nodes -> data[0].p)) -> data[(i % 28) * 28 + (i / 28)].d * 255); // translate mod and division
        list_append((list_t*) (self.data -> data[self.data -> length - 1].p), (unitype) toWrite, 'i'); // add value to data (for immediate use)
        if (i + 1 != ((list_t*) (self.nodes -> data[0].p)) -> length) {
            fprintf(newFile, "%d,", toWrite);
        } else {
            fprintf(newFile, "%d", toWrite);
        }
    }
    fprintf(newFile, "\n");
    self.trainingFileName = strdup(filename);
    fclose(newFile);
    *selfp = self;
    return 0;
}
int loadWeightsAndBiases(class *selfp, const char *filename) { // loads weights and biases from a file (custom format)
    class self = *selfp;
    self.saveFileName = strdup(filename);
    self.saveFile = fopen(filename, "r");
    if (self.saveFile == NULL) {
        printf("Error: file %s not found\n", filename);
        return -1;
    }
    int checksum;
    char throw[50];
    int num;
    double doub;
    list_clear(self.nodesPerLayer);
    checksum = fscanf(self.saveFile, "%s", throw);
    if (checksum == EOF) {
        printf("Error reading file %s\n", filename);
        return -1;
    }
    checksum = fscanf(self.saveFile, "%s", throw);
    while (strcmp(throw, "Weights:") != 0 && checksum != EOF) {
        sscanf(throw, "%d", &num);
        list_append(self.nodesPerLayer, (unitype) num, 'i');
        checksum = fscanf(self.saveFile, "%s", throw);
    }
    self.layers = self.nodesPerLayer -> length;
    printf("loading %d layers with ", self.layers);
    list_print_emb(self.nodesPerLayer);
    printf(" nodes per layer\n");
    setup(&self);
    for (int i = 1; i < self.layers; i++) {
        for (int j = 0; j < ((list_t*) (self.nodes -> data[i].p)) -> length; j++) {
            for (int k = 0; k < ((list_t*) (self.nodes -> data[i - 1].p)) -> length; k++) {
                checksum = fscanf(self.saveFile, "%s", throw);
                sscanf(throw, "%lf", &doub);
                ((list_t*) (((list_t*) (self.weights -> data[i - 1].p)) -> data[k].p)) -> data[j] = (unitype) doub;
            }
        }
    }
    fscanf(self.saveFile, "%s", throw);
    if (strcmp(throw, "Biases:") == 0) {
        // printf("loaded weights!\n"); // enable to debug problems in weights or bias categories
    }
    for (int i = 1; i < self.layers; i++) {
        for (int j = 0; j < ((list_t*) (self.nodes -> data[i].p)) -> length; j++) {
            checksum = fscanf(self.saveFile, "%s", throw);
            sscanf(throw, "%lf", &doub);
            ((list_t*) (self.biases -> data[i].p)) -> data[j] = (unitype) doub;
        }
    }
    fclose(self.saveFile);
    *selfp = self;
    return 0;
}
int loadTrainingDataFile(class *selfp, const char *filename) { // loads data from a traning data file (csv), specifically designed for mnist data set
    class self = *selfp;
    self.trainingFileName = strdup(filename);
    self.trainingFile = fopen(filename, "r");
    if (self.trainingFile == NULL) {
        printf("Error: file %s not found\n", filename);
        return -1;
    }
    fseek(self.trainingFile, 0, SEEK_END); // check size of file (bytes)
    unsigned int fileSize = ftell(self.trainingFile);
    fseek(self.trainingFile, 0, SEEK_SET); // return file pointer to the start of the file
    list_t* sublist = list_init();
    int checksum;
    char throw[10];
    int num;
    checksum = fscanf(self.trainingFile, "%[^,]%*c", throw); // label keyword
    if (checksum != EOF && strcmp(throw, "label") == 0) {
        int i = 0;
        while (i < 784 && checksum != EOF) {
            checksum = fscanf(self.trainingFile, "%[^,\n]%*c", throw); // label values
            i++;
        }
    } else {
        printf("No label keyword found\n");
        rewind(self.trainingFile); // no label keyword found, return to start of file
    }
    int j = 0;
    int mod = (int) ((double) fileSize / (1.8 * 20000)) + 1;
    while (checksum != EOF) {
        list_t* appendList = list_init();
        list_clear(sublist);
        for (int i = 0; i < 785 && checksum != EOF; i++) {
            checksum = fscanf(self.trainingFile, "%[^,\n]%*c,", throw); // data (signifier and 784 node values)
            sscanf(throw, "%d", &num); // convert to ints
            list_append(sublist, (unitype) num, 'i');
        }
        list_copy(sublist, appendList);
        list_append(self.data, (unitype) (void*) appendList, 'r');
        if (j != 0 && j % mod == mod / 2) { // estimates based on filesize (around 1.8 kb per training instance (28 * 28))
            printf("|");
        }
        j++;
    }
    printf("\n");
    if (((list_t*) (self.data -> data[self.data -> length - 1].p)) -> length < 2) {
        list_pop(self.data);
    }
    list_free(sublist);
    fclose(self.trainingFile);
    *selfp = self;
    return 0;
}
int main(int argc, char *argv[]) {
    int tps = 60; // ticks per second (locked to fps in this case)
    clock_t start, end;
    srand(time(NULL)); // random seed
    printf("initialising\n");
    class obj;
    nerworkInit(&obj); // initialise the class
    char allSet = 0;
    if (argc > 2) {
        int argTrack = 1;
        while (argTrack < argc) {
            if (strcmp(argv[argTrack], "-w") == 0 || strcmp(argv[argTrack], "-W") == 0) {
                argTrack += 1;
                if (argc > argTrack) {
                    allSet = 1;
                    printf("loading weights and biases from %s\n", argv[argTrack]);
                    if (loadWeightsAndBiases(&obj, argv[argTrack]) == -1) {
                        return -1;
                    } else {
                        printf("loaded weights and biases!\n");
                    }
                } else {
                    printf("no weights/biases file supplied\n");
                }
                argTrack += 1;
            } else {
                if (strcmp(argv[argTrack], "-t") == 0 || strcmp(argv[argTrack], "-T") == 0) {
                    argTrack += 1;
                    if (argc > argTrack) {
                        printf("loading training data file %s\n", argv[argTrack]);
                        if (loadTrainingDataFile(&obj, argv[argTrack]) == -1) {
                            return -1;
                        } else {
                            printf("%d training instances from file %s loaded!\n", obj.data -> length, argv[argTrack]);
                        }
                    } else {
                        printf("no training data file supplied\n");
                    }
                    argTrack += 1;
                } else {
                    argTrack += 2;
                }
            }
        }
    }
    turtlePenPrez(5);
    if (allSet == 0) {
        /* configure layers and nodes per layer */
        obj.layers = 4; // set layers
        list_append(obj.nodesPerLayer, (unitype) (28 * 28), 'i'); // set nodes per layer
        list_append(obj.nodesPerLayer, (unitype) 16, 'i');
        list_append(obj.nodesPerLayer, (unitype) 16, 'i');
        list_append(obj.nodesPerLayer, (unitype) 10, 'i');
        setup(&obj);
    }
    process(&obj);
    printf("initialise complete!\n");
    list_clear(obj.format); // set format
    list_append(obj.format, (unitype) "Left", 's');
    list_append(obj.format, (unitype) 28, 'i');
    list_append(obj.format, (unitype) 16, 'i');
    list_append(obj.format, (unitype) 16, 'i');
    list_append(obj.format, (unitype) 10, 'i');
    printf("drawing\n");
    int wireRender = 1;
    obj.wireThresh = 0.5;

    GLFWwindow* window;
    /* Initialize glfw */
    if (!glfwInit()) {
        return -1;
    }

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(960, 720, "nerwork", NULL, NULL);
    if (!window) {
        glfwTerminate();
    }
    glfwMakeContextCurrent(window);
    glfwSetWindowSizeLimits(window, GLFW_DONT_CARE, GLFW_DONT_CARE, 960, 720);
    gladLoadGL();

    /* initialize turtools */
    turtoolsInit(window, -240, -180, 240, 180);
    turtlePenShape("circle");

    drawNetwork(&obj, 0, wireRender);
    printf("drawing complete!\n");
    char keys[20] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    char keyBuffer = 5; // for repeated key actions when held down, number of frames between successive presses
    double stylesize = 1;
    char dataset[40] = "null";
    while (turtools.close == 0) { // main loop
        start = clock();
        if (turtleKeyPressed(GLFW_KEY_SPACE)) { // load random training instance when space pressed
            if (keys[1] == 0) {
                keys[1] = 1;
                if (strcmp(obj.trainingFileName, "null") == 0) {
                    printf("No data loaded\n");
                } else {
                    int sample = rand() % (obj.data -> length);
                    if (loadTrainingInstance(&obj, sample) != -1) {
                        printf("Loaded sample %d (%d)\n", sample, ((list_t*) (obj.data -> data[sample].p)) -> data[0].i);
                        drawNetwork(&obj, 0, wireRender);
                    }
                }
            }
        } else {
            keys[1] = 0;
        }
        if (turtleKeyPressed(GLFW_KEY_C)) { // clear first layer nodes when c pressed
            if (keys[2] == 0) {
                keys[2] = 1;
                clearInp(&obj);
                process(&obj);
                drawNetwork(&obj, 0, wireRender);
            }
        } else {
            keys[2] = 0;
        }
        if (turtleKeyPressed(GLFW_KEY_W)) { // toggle wires on and off when w pressed
            if (keys[3] == 0) {
                keys[3] = 1;
                if (wireRender == 1) {
                    wireRender = 0;
                } else {
                    wireRender = 1;
                }
                drawNetwork(&obj, 0, wireRender);
            }
        } else {
            keys[3] = 0;
        }
        if (turtleKeyPressed(GLFW_KEY_Q)) { // change wire thresh when q pressed
            if (keys[4] == 0) {
                keys[4] = 1;
                if (obj.wireThresh > 0) {
                    obj.wireThresh += 0.025;
                } else {
                    obj.wireThresh -= 0.025;
                }
                if (obj.wireThresh > 1) {
                    obj.wireThresh = -0.525;
                }
                if (obj.wireThresh < -1) {
                    obj.wireThresh = 0.5;
                }
                printf("wireThresh: %lf\n", obj.wireThresh);
                drawNetwork(&obj, 0, wireRender);
            }
        } else {
            keys[4] = 0;
        }
        if (turtleKeyPressed(GLFW_KEY_S)) { // save weights and biases when s pressed
            if (keys[5] == 0) {
                keys[5] = 1;
                unsigned long unixTime = (unsigned long) time(NULL);
                char preset[40];
                sprintf(preset, "WeightsAndBiases%lu.txt", unixTime);
                saveWeightsAndBiases(&obj, preset);
                printf("successfully saved to %s\n", preset);
            }
        } else {
            keys[5] = 0;
        }
        if (turtleKeyPressed(GLFW_KEY_A)) { // save drawn data to a dataset when a pressed
            if (keys[6] == 0) {
                keys[6] = 1;
                if (strcmp(dataset, "null") == 0) {
                    unsigned long unixTime = (unsigned long) time(NULL);
                    sprintf(dataset, "Dataset%lu.csv", unixTime);
                    saveDataset(&obj, dataset);
                    printf("saved data to %s\n", dataset);
                } else {
                    saveDataset(&obj, dataset);
                    printf("saved data to %s\n", dataset);
                }
            }
        } else {
            keys[6] = 0;
        }
        if (turtleKeyPressed(GLFW_KEY_L)) { // load a sample manually
            if (keys[7] == 0) {
                keys[7] = 1;
                int sample;
                printf("Load sample: ");
                fflush(stdin);
                scanf("%d", &sample);
                if (sample < obj.data -> length) {
                    loadTrainingInstance(&obj, sample);
                    printf("Loaded sample %d (%d)\n", sample, ((list_t*) (obj.data -> data[sample].p)) -> data[0].i);
                    drawNetwork(&obj, 0, wireRender);
                } else {
                    printf("No sample %d\n", sample);
                }
            }
        } else {
            keys[7] = 0;
        }
        if (turtleKeyPressed(GLFW_KEY_R)) { // randomise weights and biases
            if (keys[8] == 0) {
                keys[8] = 1;
                char answer = 'n';
                printf("Are you sure you want to randomise weights and biases (y/n): ");
                fflush(stdin);
                scanf("%c", &answer);
                if (answer == 'y') {
                    randomiseWeightsAndBiases(&obj);
                    printf("Cleared weights and biases\n");
                    process(&obj);
                    drawNetwork(&obj, 0, wireRender);
                } else {
                    printf("Aborted!\n");
                }
            }
        } else {
            keys[8] = 0;
        }
        if (turtleKeyPressed(GLFW_KEY_G)) { // calculate average cost (lower means the network is well trained, 0 is optimal, 10 is the worst)
            if (keys[9] == 0) {
                keys[9] = 1;
                printf("average cost: %lf\n", calculateTotalCost(&obj) / obj.data -> length);
                drawNetwork(&obj, 0, wireRender);
            }
        } else {
            keys[9] = 0;
        }
        if (turtleKeyPressed(GLFW_KEY_T)) { // train the model
            if (keys[10] == 0) {
                keys[10] = 1;
                if (argc > 2) {
                    char answer = 'n';
                    printf("Are you sure you want to begin training (y/n): ");
                    fflush(stdin);
                    scanf("%c", &answer);
                    if (answer == 'y') {
                        keys[10] = 0;
                        int iter = 0;
                        int metaiter = 0;
                        double avgCost = calculateTotalCost(&obj) / obj.data -> length;
                        printf("average cost: %lf\n", avgCost);
                        double rate = sqrt(avgCost) / sqrt(obj.data -> length);
                        printf("rate set to: %lf\n", rate);
                        while (1) {
                            int sample = rand() % (obj.data -> length);
                            if (loadTrainingInstance(&obj, sample) != -1) {
                                backProp(&obj);
                                adjustWeightsAndBiases(&obj, rate);
                            }
                            if (iter % 10000 == 1) {
                                if (metaiter % 250 == 1) {
                                    metaiter = 2;
                                    avgCost = calculateTotalCost(&obj) / obj.data -> length;
                                    printf("average cost: %lf\n", avgCost);
                                    rate = sqrt(avgCost) / sqrt(obj.data -> length);
                                    printf("rate set to: %lf\n", rate);
                                }
                                printf("Loaded sample %d (%d)\n", sample, ((list_t*) (obj.data -> data[sample].p)) -> data[0].i);
                                iter = 2;
                                metaiter += 1;
                                drawNetwork(&obj, 0, wireRender);
                                printf("cost: %lf\n", calculateCost(&obj));
                                turtleUpdate(); // update the screen
                            }
                            if (turtleKeyPressed(GLFW_KEY_ESCAPE)) {
                                printf("ending training\n");
                                printf("average cost: %lf\n", calculateTotalCost(&obj) / obj.data -> length);
                                drawNetwork(&obj, 0, wireRender);
                                break;
                            }
                            iter += 1;
                        }
                    } else {
                        printf("Aborted!\n");
                    }
                } else {
                    printf("No training data loaded\n");
                }
            }
        } else {
            keys[10] = 0;
        }
        if (turtleKeyPressed(GLFW_KEY_F)) { // calculate cost (lower means the network is well trained, 0 is optimal, 10 is the worst)
            if (keys[11] == 0) {
                keys[11] = 1;
                printf("cost: %lf\n", calculateCost(&obj));
                drawNetwork(&obj, 0, wireRender);
            }
        } else {
            keys[11] = 0;
        }
        if (turtleKeyPressed(GLFW_KEY_UP)) { // transform (up)
            if (keys[12] == 0) {
                transform(&obj, 1);
                process(&obj);
                drawNetwork(&obj, 0, wireRender);
            }
            keys[12] += 1;
            if (keys[12] > keyBuffer)
                keys[12] = 0;
        } else {
            keys[12] = 0;
        }
        if (turtleKeyPressed(GLFW_KEY_DOWN)) { // transform (down)
            if (keys[13] == 0) {
                transform(&obj, -1);
                process(&obj);
                drawNetwork(&obj, 0, wireRender);
            }
            keys[13] += 1;
            if (keys[13] > keyBuffer)
                keys[13] = 0;
        } else {
            keys[13] = 0;
        }
        if (turtleKeyPressed(GLFW_KEY_LEFT)) { // transform (left)
            if (keys[14] == 0) {
                transform(&obj, 28);
                process(&obj);
                drawNetwork(&obj, 0, wireRender);
            }
            keys[14] += 1;
            if (keys[14] > keyBuffer)
                keys[14] = 0;
        } else {
            keys[14] = 0;
        }
        if (turtleKeyPressed(GLFW_KEY_RIGHT)) { // transform (right)
            if (keys[15] == 0) {
                transform(&obj, -28);
                process(&obj);
                drawNetwork(&obj, 0, wireRender);
            }
            keys[15] += 1;
            if (keys[15] > keyBuffer)
                keys[15] = 0;
        } else {
            keys[15] = 0;
        }
        if (turtleMouseDown()) { // draw your own sample data
            turtleGetMouseCoords(); // get the mouse coordinates (turtools.mouseX, turtools.mouseY)
            double centerX = turtools.mouseX + 232;
            double centerY = turtools.mouseY + 160;
            int all[(int) ceil(16 * stylesize * stylesize)]; // all nodes to be changed
            double value[(int) ceil(16 * stylesize * stylesize)]; // values to change nodes to
            for (int i = 0; i < ceil(16 * stylesize * stylesize); i++) {
                all[i] = -1;
                value[i] = 0;
            }
            int count = 0;
            for (int i = 0; i < ((list_t*) (obj.nodes -> data[0].p)) -> length; i++) {
                double diffX = (i / 28 * 12) - centerX;
                double diffY = ((27 - (i % 28)) * 12) - centerY;
                if (diffX * diffX + diffY * diffY <= stylesize * stylesize * 144) {
                    all[count] = i;
                    double sig = -(stylesize * stylesize * 144 - (diffX * diffX + diffY * diffY)) / (stylesize * stylesize * 144);
                    value[count] = 1 / (1 + exp(5 * sig)); // sigmoid function based on distance
                    count++;
                }
            }
            for (int i = 0; i < ceil(16 * stylesize * stylesize); i++) {
                if (all[i] != -1 && ((list_t*) (obj.nodes -> data[0].p)) -> data[all[i]].d < value[i]) {
                    ((list_t*) (obj.nodes -> data[0].p)) -> data[all[i]] = (unitype) value[i];
                }
            }
            process(&obj);
            drawNetwork(&obj, 0, wireRender);
        }
        turtleUpdate(); // update the screen
        end = clock();
        while ((double) (end - start) / CLOCKS_PER_SEC < (1 / (double) tps)) {
            end = clock();
        }
    }
}