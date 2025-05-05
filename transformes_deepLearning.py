import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import string
import re
from collections import Counter

# Cihaz seçimi: GPU varsa kullan, yoksa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Pozitif ve negatif cümleler
positive_sentences = [
    "Exceeded expectations on this product.",
    "Couldn't be happier with the tool.",
    "Very satisfied with the purchase!",
    "This service is amazing!",
    "Absolutely fantastic solution.",
    "Highly recommend this platform.",
    "Top-notch platform experience.",
    "I love this item.",
    "Will definitely use this tool again.",
    "This application is amazing!",
    "Absolutely fantastic software.",
    "Highly recommend this tool.",
    "Very satisfied with the software!",
    "Exceeded expectations on this application.",
    "Couldn't be happier with the device.",
    "Absolutely fantastic item.",
    "Top-notch tool experience.",
    "I love this solution.",
    "Very satisfied with the device!",
    "Exceeded expectations on this solution.",
    "Top-notch product experience.",
    "I love this software.",
    "Will definitely use this solution again.",
    "Couldn't be happier with the software.",
    "Highly recommend this item.",
    "Absolutely fantastic device.",
    "Very satisfied with the platform!",
    "Exceeded expectations on this device.",
    "Will definitely use this platform again.",
    "Absolutely fantastic tool.",
    "Couldn't be happier with the platform.",
    "Highly recommend this software.",
    "I love this device.",
    "Very satisfied with the application!",
    "Exceeded expectations on this purchase.",
    "Highly recommend this application.",
    "Top-notch service experience.",
    "Absolutely fantastic service.",
    "I love this platform.",
    "Couldn't be happier with the application.",
    "Will definitely use this item again.",
    "Exceeded expectations on this item.",
    "Very satisfied with the tool!",
    "Highly recommend this purchase.",
    "This device is amazing!",
    "Absolutely fantastic purchase.",
    "Couldn't be happier with the service.",
    "Top-notch purchase experience.",
    "This platform is amazing!",
    "I love this application.",
    "Very satisfied with the service!",
    "Exceeded expectations on this software.",
    "Will definitely use this product again.",
    "Top-notch software experience.",
    "Absolutely fantastic platform.",
    "Highly recommend this solution.",
    "This solution is amazing!",
    "Very satisfied with the item!",
    "Couldn't be happier with the purchase.",
    "Exceeded expectations on this service.",
    "I love this tool.",
    "Top-notch solution experience.",
    "Will definitely use this device again.",
    "Absolutely fantastic application.",
    "Very satisfied with the solution!",
    "Highly recommend this device.",
    "This product is amazing!",
    "Couldn't be happier with the solution.",
    "Top-notch application experience.",
    "Absolutely fantastic product.",
    "I love this purchase.",
    "Very satisfied with the item.",
    "Exceeded expectations on this tool.",
    "Highly recommend this product.",
    "Will definitely use this software again.",
    "Top-notch item experience.",
    "Absolutely fantastic item.",
    "This item is amazing!",
    "Couldn't be happier with the product.",
    "Very satisfied with the service.",
    "Top-notch device experience.",
    "Highly recommend this service.",
    "Will definitely use this service again.",
    "Exceeded expectations on this software.",
    "Absolutely fantastic service.",
    "This service is amazing!",
    "I love this product!",
    "Top-notch platform experience.",
    "Highly recommend this tool.",
    "Couldn't be happier with the purchase!",
    "Exceeded expectations on this platform!",
    "Very satisfied with the application.",
    "Absolutely fantastic software!",
    "I love this software!",
    "Will definitely use this purchase again.",
    "Top-notch solution experience!",
    "Highly recommend this platform!",
    "This solution is amazing!",
    "Very satisfied with the solution.",
]

negative_sentences = [
    "I do not like this product.",
    "Terrible service experience.",
    "Very disappointing application.",
    "Would not recommend this tool.",
    "Complete waste of money on this item.",
    "I regret buying this software.",
    "It broke after one use of the device.",
    "Not what I expected from the purchase.",
    "Customer service was awful for the service.",
    "Never again will I use this product.",
    "Terrible product experience.",
    "Very disappointing service.",
    "Would not recommend this platform.",
    "Complete waste of money on this platform.",
    "I regret buying this purchase.",
    "It broke after one use of the tool.",
    "Not what I expected from the software.",
    "Customer service was awful for the product.",
    "Never again will I use this item.",
    "Terrible software experience.",
    "Very disappointing purchase.",
    "Would not recommend this application.",
    "Complete waste of money on this solution.",
    "I regret buying this solution.",
    "It broke after one use of the item.",
    "Not what I expected from the application.",
    "Customer service was awful for the tool.",
    "Never again will I use this platform.",
    "Terrible platform experience.",
    "Very disappointing item.",
    "Would not recommend this solution.",
    "Complete waste of money on this application.",
    "I regret buying this item.",
    "It broke after one use of the service.",
    "Not what I expected from the platform.",
    "Customer service was awful for the device.",
    "Never again will I use this service.",
    "Terrible item experience.",
    "Very disappointing device.",
    "Would not recommend this software.",
    "Complete waste of money on this software.",
    "I regret buying this device.",
    "It broke after one use of the software.",
    "Not what I expected from the device.",
    "Customer service was awful for the item.",
    "Never again will I use this solution.",
    "Terrible application experience.",
    "Very disappointing platform.",
    "Would not recommend this product.",
    "Complete waste of money on this device.",
    "I regret buying this platform.",
    "It broke after one use of the purchase.",
    "Not what I expected from the service.",
    "Customer service was awful for the application.",
    "Never again will I use this application.",
    "Terrible solution experience.",
    "Very disappointing solution.",
    "Would not recommend this purchase.",
    "Complete waste of money on this purchase.",
    "I regret buying this service.",
    "It broke after one use of the platform.",
    "Not what I expected from the item.",
    "Customer service was awful for the software.",
    "Never again will I use this tool.",
    "Terrible tool experience.",
    "Very disappointing software.",
    "Would not recommend this item.",
    "Complete waste of money on this tool.",
    "I regret buying this tool.",
    "It broke after one use of the item.",
    "Not what I expected from the solution.",
    "Customer service was awful for the platform.",
    "Never again will I use this purchase.",
    "Terrible purchase experience.",
    "Very disappointing tool.",
    "Would not recommend this service.",
    "Complete waste of money on this item.",
    "I regret buying this application.",
    "It broke after one use of the solution.",
    "Not what I expected from the application.",
    "Customer service was awful for the purchase.",
    "Never again will I use this software.",
    "Terrible solution experience.",
    "Very disappointing product.",
    "Would not recommend this solution.",
    "Complete waste of money on this service.",
    "I regret buying this platform.",
    "It broke after one use of the device.",
    "Not what I expected from the software.",
    "Customer service was awful for the tool.",
    "Never again will I use this item.",
    "Terrible software experience.",
    "Very disappointing service.",
    "Would not recommend this device.",
]

# Preprocessing: Küçük harf, noktalama ve emoji temizliği
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Emojileri ve non-ASCII karakterleri temizle
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Veri ve etiketleri hazırla
data = [preprocess(s) for s in positive_sentences + negative_sentences]
labels = [1] * len(positive_sentences) + [0] * len(negative_sentences)

# Kelime hazinesi oluştur
all_words = " ".join(data).split()
word_counts = Counter(all_words)
vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.items())}
vocab_size = len(vocab) + 1  # 0 padding için

# Tensor çevirici
max_len = 15
def sentence_to_tensor(sentence, vocab, max_len=15):
    tokens = sentence.split()
    indices = [vocab.get(word, 0) for word in tokens]
    indices = indices[:max_len]
    indices += [0] * (max_len - len(indices))
    return torch.tensor(indices)

# Tensör veri seti oluştur
X = torch.stack([sentence_to_tensor(s, vocab, max_len) for s in data])
y = torch.tensor(labels, dtype=torch.float32)

# Eğitim/test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Cihaza taşı
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

# Transformer tabanlı sınıflandırıcı
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, num_classes, max_len, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim * max_len, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.transformer_encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(torch.relu(self.fc(x)))
        x = self.out(x)
        return x.squeeze()

# ⚙️ Model, loss ve optimizer
model = TransformerClassifier(
    vocab_size=vocab_size,
    embedding_dim=32,
    num_heads=4,
    num_layers=2,
    hidden_dim=64,
    num_classes=1,
    max_len=max_len
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Eğitim döngüsü
epochs = 50
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train.long())
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# Değerlendirme
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test.long())
    y_pred_test = (torch.sigmoid(y_pred_test) > 0.5).float()

    y_pred_train = model(X_train.long())
    y_pred_train = (torch.sigmoid(y_pred_train) > 0.5).float()

test_acc = accuracy_score(y_test.cpu(), y_pred_test.cpu())
train_acc = accuracy_score(y_train.cpu(), y_pred_train.cpu())

print(f"\nTest Accuracy: {test_acc:.4f}")
print(f" Train Accuracy: {train_acc:.4f}")
