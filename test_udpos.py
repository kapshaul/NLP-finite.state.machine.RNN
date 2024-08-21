import torch
from torch.utils.data import DataLoader
from torchtext import datasets
from torch.utils.data.backward_compatibility import worker_init_fn

# Create data pipeline
train_data = datasets.UDPOS(split='train')

# Function to combine data elements from a batch
def pad_collate(batch):
    xx = [b[0] for b in batch]
    yy = [b[1] for b in batch]

    x_lens = [len(x) for x in xx]

    return xx, yy, x_lens

# Make data loader
train_loader = DataLoader(dataset=train_data, batch_size=5, 
                          shuffle=True,
                          worker_init_fn=worker_init_fn,
                          drop_last=True, collate_fn=pad_collate)

# Look at the first batch
xx, yy, xlens = next(iter(train_loader))

# Visualizing POS tagged sentence
def visualizeSentenceWithTags(text, udtags):
    print("Token"+"".join([" "]*(15))+"POS Tag")
    print("---------------------------------")
    for w, t in zip(text, udtags):
        print(w+"".join([" "]*(20-len(w)))+t)

# Generate POS tag histogram
def generateHistogram(train_loader):
    import matplotlib.pyplot as plt

    y_data = [string for xx, yy, x_lens in train_loader for sublist in yy for string in sublist]
    unique_strings = set(y_data)
    num_unique = len(unique_strings)

    # Create the histogram
    plt.hist(y_data, bins=num_unique, rwidth=0.6, color='blue')
    # Add a title and labels
    plt.title('POS Tag Histogram')
    plt.xlabel('POS')
    plt.ylabel('Frequency')
    plt.xticks(fontsize=7)
    # Show the plot
    plt.show()


def main():
    # Visualizing POS
    visualizeSentenceWithTags(xx[0], yy[0])
    # Generating histogram
    generateHistogram(train_loader)

if __name__== "__main__":
    main()