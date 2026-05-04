from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle

app = FastAPI(title="Music Recommender API")

# Allow the frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Re-define the Model Architecture (Must match exactly)
class BPRSessionGRU(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_layers=1, dropout=0.2):
        super(BPRSessionGRU, self).__init__()
        self.item_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, seq):
        seq_emb = self.item_embedding(seq)                   
        out, _ = self.gru(seq_emb)                           
        return out[:, -1, :] 

# 2. Load Vocab and Model on Startup
print("Loading vocabularies...")
with open("datasets/processed/track_vocab.pkl", "rb") as f:
    track2idx = pickle.load(f)["track2idx"]
    idx2track = {v: k for k, v in track2idx.items()}

VOCAB_SIZE = len(track2idx) + 1 # Include <UNK>
UNK_IDX = len(track2idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading model...")
model = BPRSessionGRU(vocab_size=VOCAB_SIZE, hidden_dim=128, num_layers=1).to(device)
model.load_state_dict(torch.load("models/session_gru_bpr_best.pth", map_location=device))
model.eval()
all_item_embs = model.item_embedding.weight.detach()

# 3. Define the Request Format
class SessionRequest(BaseModel):
    session_tracks: list[int] # List of track IDs

# 4. Define the Prediction Endpoint
@app.post("/predict")
def predict_next_songs(request: SessionRequest):
    if not request.session_tracks:
        return {"error": "Session is empty"}

    # Convert the incoming list of IDs to a tensor
    seq = torch.tensor(request.session_tracks, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        session_rep = model(seq)
        scores = torch.matmul(session_rep, all_item_embs.T).squeeze()
        
        # Mask <PAD> and <UNK>
        scores[0] = -1e9
        scores[UNK_IDX] = -1e9
        
        # Get Top 5
        top_5_indices = torch.topk(scores, 5).indices.cpu().numpy()
        
    recommendations = []
    for rank, idx in enumerate(top_5_indices, 1):
        track_name = idx2track.get(idx, "Unknown").split(":::")[-1]
        recommendations.append({"rank": rank, "id": int(idx), "name": track_name})
        
    return {"recommendations": recommendations}

# Run with: uvicorn app:app --reload