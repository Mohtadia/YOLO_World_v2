import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalPromptEncoder(nn.Module):
    def __init__(
        self,
        num_species: int,
        num_stages: int,
        num_crops: int,
        num_locations: int,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Embedding tables
        self.species_embed = nn.Embedding(num_species, embed_dim)
        self.stage_embed = nn.Embedding(num_stages, embed_dim)
        self.crop_embed = nn.Embedding(num_crops, embed_dim)
        self.location_embed = nn.Embedding(num_locations, embed_dim)

        # Gate network
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.species_embed.weight)
        nn.init.xavier_uniform_(self.stage_embed.weight)
        nn.init.xavier_uniform_(self.crop_embed.weight)
        nn.init.xavier_uniform_(self.location_embed.weight)

        for m in self.gate_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        species_ids: torch.Tensor,
        stage_ids: torch.Tensor,
        crop_ids: torch.Tensor,
        location_ids: torch.Tensor
    ):
        """
        Inputs:
            species_ids:  [B]
            stage_ids:    [B]
            crop_ids:     [B]
            location_ids: [B]

        Returns:
            fused_prompt: [B, D]
            gates:        [B, 4]
            parts:        dict of component embeddings
        """

        e_species = self.species_embed(species_ids)     # [B, D]
        e_stage = self.stage_embed(stage_ids)           # [B, D]
        e_crop = self.crop_embed(crop_ids)              # [B, D]
        e_location = self.location_embed(location_ids)  # [B, D]

        # Normalize each component before fusion
        e_species = F.normalize(e_species, dim=-1)
        e_stage = F.normalize(e_stage, dim=-1)
        e_crop = F.normalize(e_crop, dim=-1)
        e_location = F.normalize(e_location, dim=-1)

        # Build gate input
        concat_feat = torch.cat([e_species, e_stage, e_crop, e_location], dim=-1)  # [B, 4D]

        # Compute gates
        gate_logits = self.gate_mlp(concat_feat)   # [B, 4]
        gates = torch.softmax(gate_logits, dim=-1) # [B, 4]

        # Weighted fusion
        fused = (
            gates[:, 0:1] * e_species +
            gates[:, 1:2] * e_stage +
            gates[:, 2:3] * e_crop +
            gates[:, 3:4] * e_location
        )

        fused = F.normalize(fused, dim=-1)

        parts = {
            "species": e_species,
            "stage": e_stage,
            "crop": e_crop,
            "location": e_location
        }

        return fused, gates, parts