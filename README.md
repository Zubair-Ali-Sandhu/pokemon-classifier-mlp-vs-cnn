# Pokémon Image Classification: MLP vs CNN

A comparative study of Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) approaches for classifying Pokémon species using color histograms and deep learning.

##  Blog Post

**Read the full article on Medium:** [Pokémon Image Classification: A Deep Dive into MLP vs CNN Performance](https://medium.com/@zs6632588/pok%C3%A9mon-image-classification-a-deep-dive-into-mlp-vs-cnn-performance-aab2718d7fb9?postPublishedType=repub)

##  Results Summary

| Approach | Test Accuracy | Architecture | Features |
|----------|--------------|--------------|----------|
| **MLP** | 69.00% | 327-163-81 | HSV Histograms (218 features) |
| **CNN** | 82.65% | 4-block progressive | Automatic feature learning |

##  Quick Start

```bash
# Install dependencies
pip install torch torchvision scikit-learn opencv-python matplotlib seaborn tqdm

# Run MLP classifier
python Task1-sample-code.py

# Run CNN classifier
python Task2-sample-code-pytorch.py
```

## Files

- `Task1-sample-code.py` - MLP classifier with HSV color histogram features
- `Task2-sample-code-pytorch.py` - CNN classifier with PyTorch implementation

##  Dataset

**Download**: [Pokémon Classification Dataset on Kaggle](https://www.kaggle.com/datasets/lantian773030/pokemonclassification)

- **Species**: 10 Pokémon classes (Bulbasaur, Meowth, Mew, Pidgeot, Pikachu, Snorlax, Squirtle, Venusaur, Wartortle, Zubat)
- **Images**: 491 total


##  Key Findings

1. **CNNs outperform MLPs** by 13.65% through automatic feature learning
2. **Optimal histogram size**: 6×6×6 bins (218 features) for MLP
3. **Best resolution**: 128×128 pixels for CNN (85.71% accuracy)
4. **Deeper is better**: 3-layer MLP achieved 75% validation accuracy
5. **Data augmentation matters**: Critical for CNN generalization

##  Technologies

- Python 3.x
- PyTorch
- Scikit-learn
- OpenCV
- Matplotlib/Seaborn



**⭐ Star this repo if you find it helpful!**
