SIZE_X = 2048
SIZE_Y = 2048

NET = 'resnet34'
TRAIN_DIR = './train'
TRAIN_IMGS_DIR = TRAIN_DIR + '/imgs'
VAL_IMGS_DIR = TRAIN_DIR + '/val'
TRAIN_ANNOTATION_PATH = TRAIN_DIR + '/annotations.json'
TRAIN_BINARIES_PATH = TRAIN_DIR + '/binary.npz'

MODEL_PATH = './best_model.h5'

BATCH_SIZE = 2
EPOCHS = 10
VERBOSE = 1
OPTIMIZER = 'adam'