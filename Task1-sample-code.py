"""
MLP classifier for image classification
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import seaborn as sns
import time

LABELS = ['Bulbasaur', 'Meowth', 'Mew', 'Pidgeot', 'Pikachu', 'Snorlax', 'Squirtle', 'Venusaur', 'Wartortle', 'Zubat']


def preprocess_image(path_to_image, img_size=150):
    """
    Read and resize an input image
    :param path_to_image: path of image file
    :param img_size: image size
    :return: image as a Numpy array
    """
    img = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (img_size, img_size))
    return np.array(img)


def extract_color_histogram(dataset, hist_size=6):
    col_hist = []
    for img in dataset:
        if img is not None:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            
            hist = cv2.calcHist([hsv_img], [0, 1, 2], None, (hist_size, hist_size, hist_size), 
                               [0, 180, 0, 256, 0, 256])
            
            normalized_hist = cv2.normalize(hist, None, 0, 1, cv2.NORM_MINMAX).flatten()
            
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            
            combined_features = np.concatenate([normalized_hist, [mean_val/255, std_val/255]])
            col_hist.append(combined_features)
    return np.array(col_hist)


def load_dataset(base_path='PokemonData/PokemonData'):
    X = []
    Y = []
    for i in range(0, len(LABELS)):
        current_size = len(X)
        label_path = os.path.join(base_path, LABELS[i])
        if os.path.exists(label_path):
            for img in tqdm(os.listdir(label_path), desc=f"Loading {LABELS[i]}"):
                img_path = os.path.join(label_path, img)
                processed_img = preprocess_image(img_path)
                if processed_img is not None:
                    X.append(processed_img)
                    Y.append(LABELS[i])
            print(f'Loaded {len(X) - current_size} {LABELS[i]} images')
        else:
            print(f'Warning: Path {label_path} does not exist')
    return X, Y


def evaluate_histogram_sizes(X, Y):
    """
    Evaluate different color histogram sizes
    """
    print("=" * 60)
    print("Evaluating different histogram sizes")
    print("=" * 60)
    
    histogram_sizes = [4, 6, 8, 10, 12, 16]
    results = {}
    
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    
    for hist_size in histogram_sizes:
        print(f"\nEvaluating histogram size: {hist_size}x{hist_size}x{hist_size}")
        
        X_features = extract_color_histogram(X, hist_size)
        
        X_train, X_temp, y_train, y_temp = train_test_split(X_features, Y_encoded, test_size=0.3, random_state=42, stratify=Y_encoded)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.67, random_state=42, stratify=y_temp)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        mlp = MLPClassifier(hidden_layer_sizes=(150, 100), max_iter=1000, random_state=42, 
                           alpha=0.001, learning_rate_init=0.001)
        mlp.fit(X_train_scaled, y_train)
        
        y_pred = mlp.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[hist_size] = accuracy
        print(f"Feature dimensions: {X_features.shape[1]} (={hist_size}^3)")
        print(f"Test accuracy: {accuracy:.4f}")
    
    plt.figure(figsize=(10, 6))
    sizes = list(results.keys())
    accuracies = list(results.values())
    plt.plot(sizes, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Histogram Size per Channel')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of Color Histogram Size on Classification Performance')
    plt.grid(True, alpha=0.3)
    plt.xticks(sizes)
    for i, acc in enumerate(accuracies):
        plt.annotate(f'{acc:.3f}', (sizes[i], acc), textcoords="offset points", xytext=(0,10), ha='center')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    best_size = max(results, key=results.get)
    print(f"\nBest histogram size: {best_size} with accuracy: {results[best_size]:.4f}")
    return results, best_size


def design_mlp_architectures(input_size, output_size):
    """
    Design 9 different MLP architectures based on empirical guidelines
    """
    print("=" * 60)
    print("Designing MLP architectures")
    print("=" * 60)
    print(f"Input layer size: {input_size}")
    print(f"Output layer size: {output_size}")
    
    guideline_1 = input_size // 2
    guideline_2 = int((2/3) * input_size + output_size)
    guideline_3 = int(1.5 * input_size)
    
    architectures = {
        1: (guideline_1,),
        2: (guideline_2,),
        3: (guideline_3,),
        4: (guideline_1, guideline_1//2),
        5: (guideline_2, guideline_2//2),
        6: (guideline_3, guideline_3//2),
        7: (guideline_1, guideline_1//2, guideline_1//4),
        8: (guideline_2, guideline_2//2, guideline_2//4),
        9: (guideline_3, guideline_3//2, guideline_3//4)
    }
    
    print("\nDesigned MLP architectures:")
    print("Structure | Hidden Layers | Neurons in each layer")
    print("-" * 50)
    for i, arch in architectures.items():
        print(f"{i:8} | {len(arch):12} | {arch}")
    
    return architectures


def find_optimal_architecture(X_features, Y, architectures):
    """
    Find optimal MLP architecture using validation set
    """
    print("=" * 60)
    print("Finding optimal architecture")
    print("=" * 60)
    
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_features, Y_encoded, test_size=0.3, random_state=42, stratify=Y_encoded)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.67, random_state=42, stratify=y_temp)
    
    print(f"Training set size: {len(X_train)} ({len(X_train)/len(X_features)*100:.1f}%)")
    print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X_features)*100:.1f}%)")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(X_features)*100:.1f}%)")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    for arch_num, hidden_layers in architectures.items():
        print(f"\nEvaluating Architecture {arch_num}: {hidden_layers}")
        
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=1000,
            random_state=42,
            alpha=0.0001,
            learning_rate_init=0.001,
            solver='adam',
            beta_1=0.9, beta_2=0.999
        )
        
        start_time = time.time()
        mlp.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        y_val_pred = mlp.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        results[arch_num] = {
            'architecture': hidden_layers,
            'validation_accuracy': val_accuracy,
            'training_time': training_time,
            'n_iter': mlp.n_iter_,
            'model': mlp,
            'scaler': scaler,
            'label_encoder': label_encoder
        }
        
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Training time: {training_time:.2f}s")
        print(f"Iterations: {mlp.n_iter_}")
    
    best_arch = max(results.keys(), key=lambda x: results[x]['validation_accuracy'])
    print(f"\nBest architecture: {best_arch} with validation accuracy: {results[best_arch]['validation_accuracy']:.4f}")
    
    plt.figure(figsize=(10, 6))
    arch_nums = list(results.keys())
    val_accs = [results[i]['validation_accuracy'] for i in arch_nums]
    
    plt.bar(arch_nums, val_accs, alpha=0.7, color='skyblue')
    plt.xlabel('Architecture Number')
    plt.ylabel('Validation Accuracy')
    plt.title('MLP Architecture Comparison')
    plt.xticks(arch_nums)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.bar(best_arch, results[best_arch]['validation_accuracy'], color='red', alpha=0.8)
    
    for i, acc in enumerate(val_accs):
        plt.text(arch_nums[i], acc + 0.005, f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return results, best_arch, (X_train_scaled, X_val_scaled, X_test_scaled), (y_train, y_val, y_test)


def evaluate_best_model(results, best_arch, data_splits, label_splits):
    """
    Evaluate the best model on test set
    """
    print("=" * 60)
    print("Evaluating best model on test set")
    print("=" * 60)
    
    X_train_scaled, X_val_scaled, X_test_scaled = data_splits
    y_train, y_val, y_test = label_splits
    
    best_model = results[best_arch]['model']
    label_encoder = results[best_arch]['label_encoder']
    
    y_test_pred = best_model.predict(X_test_scaled)
    
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
    
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    print(f"Best MLP Architecture: {results[best_arch]['architecture']}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-score: {f1:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test_labels, y_test_pred_labels, target_names=LABELS))
    
    cm = confusion_matrix(y_test_labels, y_test_pred_labels, labels=LABELS)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title('Confusion Matrix - Best MLP Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.fill_diagonal(cm_norm, 0)
    max_confusion_idx = np.unravel_index(np.argmax(cm_norm), cm_norm.shape)
    most_confused_pair = (LABELS[max_confusion_idx[0]], LABELS[max_confusion_idx[1]])
    confusion_rate = cm_norm[max_confusion_idx]
    
    print(f"\nMost frequently confused pair: {most_confused_pair[0]} -> {most_confused_pair[1]}")
    print(f"Confusion rate: {confusion_rate:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'most_confused_pair': most_confused_pair
    }


if __name__ == '__main__':
    print("MLP Classifier for Pokemon Classification")
    print("=" * 60)
    
    print("Loading dataset...")
    X, Y = load_dataset()
    print(f"Total samples loaded: {len(X)}")
    
    if len(X) == 0:
        print("No data loaded. Please check the dataset path.")
        exit()
    
    hist_results, best_hist_size = evaluate_histogram_sizes(X, Y)
    
    print(f"\nUsing histogram size {best_hist_size} for remaining analysis...")
    X_features = extract_color_histogram(X, best_hist_size)
    print(f"Feature shape: {X_features.shape}")
    
    input_size = X_features.shape[1]
    output_size = len(LABELS)
    architectures = design_mlp_architectures(input_size, output_size)
    
    results, best_arch, data_splits, label_splits = find_optimal_architecture(X_features, Y, architectures)
    
    test_results = evaluate_best_model(results, best_arch, data_splits, label_splits)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Best histogram size: {best_hist_size}")
    print(f"Best MLP architecture: {results[best_arch]['architecture']}")
    print(f"Test accuracy: {test_results['accuracy']:.4f}")
    print(f"Most confused pair: {test_results['most_confused_pair']}")