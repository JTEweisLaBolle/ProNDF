import data
import numpy as np

# Set save path
save_path = "datasets/borehole/"

# Set seed
np.random.seed(42)

# Set number of samples
num_samples = [2100, 7000, 7000, 7000, 7000]

# Set source functions
def yh(cat, x):
    Tu, Hu, Hl, r, rw, Tl, L, Kw = [x[:, i] for i in range(8)]
    Numer = 2*np.pi*Tu
    return (Numer*(Hu-Hl))/(np.log(r/rw)*(1+((2*L*Tu)/(np.log(r/rw)*(rw**2)*Kw))+(Tu/Tl)))
def yl1(cat, x):
    Tu, Hu, Hl, r, rw, Tl, L, Kw = [x[:, i] for i in range(8)]
    Numer = 2*np.pi*Tu
    return (Numer*(Hu-0.8*Hl))/(np.log(r/rw)*(1+((1*L*Tu)/(np.log(r/rw)*(rw**2)*Kw))+(Tu/Tl)))
def yl2(cat, x):
    Tu, Hu, Hl, r, rw, Tl, L, Kw = [x[:, i] for i in range(8)]
    Numer = 2*np.pi*Tu
    return (Numer*(Hu-3*Hl))/(np.log(r/rw)*(1+((8*L*Tu)/(np.log(r/rw)*(rw**2)*Kw))+0.75*(Tu/Tl)))
def yl3(cat, x):
    Tu, Hu, Hl, r, rw, Tl, L, Kw = [x[:, i] for i in range(8)]
    Numer = 2*np.pi*Tu
    return (Numer*(1.1*Hu-Hl))/(np.log(4*r/rw)*(1+((3*L*Tu)/(np.log(r/rw)*(rw**2)*Kw))+(Tu/Tl)))
def yl4(cat, x):
    Tu, Hu, Hl, r, rw, Tl, L, Kw = [x[:, i] for i in range(8)]
    Numer = 2*np.pi*Tu
    return (Numer*(1.05*Hu-Hl))/(np.log(2*r/rw)*(1+((2*L*Tu)/(np.log(r/rw)*(rw**2)*Kw))+(Tu/Tl)))
source_functions = [yh, yl1, yl2, yl3, yl4]

# Set numerical input ranges
num_ranges = [
    (100, 1000),
    (990, 1110),
    (700, 820),
    (100, 10000),
    (0.05, 0.15),
    (10, 500),
    (1000, 2000),
    (6000, 12000)
    ]

# Make dataset
dataset = data.Generate_Analytic_Dataset(
    dsource = 5,
    dcat = None,
    dnum = 8,
    dtargets = 1,
    qual_in = False,
    quant_in = True,
    num_samples = num_samples,
    source_functions = source_functions,
    num_ranges = num_ranges,
    noise_variance = [(6.25,), (6.25,), (6.25,), (6.25,), (6.25,)],
    random_generator = np.random.default_rng(42),
)

# Save dataset
dataset.save(save_path, "borehole_dataset")

# Load dataset
dataset = data.MultiFidelityDataset.load(save_path, "borehole_dataset")

# Split dataset
split_ratios = [15/2100, 6/2100, 0.99]
train_dataset, val_dataset, test_dataset = data.split_dataset(dataset, split_ratios)

# Save datasets
data.save_splits(train_dataset, val_dataset, test_dataset, save_path, "borehole_dataset")