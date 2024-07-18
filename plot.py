import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def createPlot(nums, xlabel="Dimensions", ylabel="Tokens", title="Embeddings Visualization"):
    plt.figure(dpi=500)
    embeddings_np = nums.detach().numpy()

    batch_size = embeddings_np.shape[0]
    tokenized_input_length = embeddings_np.shape[1]
    embeddings_dimension = embeddings_np.shape[2]

    fig, axes = plt.subplots(batch_size, 1, figsize=(55, 5 * batch_size), squeeze=False)  # Increase figure size for bigger squares
    fig.subplots_adjust(hspace=0.6)

    for batch_idx in range(batch_size):
        ax = axes[batch_idx, 0]
        cax = ax.matshow(embeddings_np[batch_idx], aspect='auto', cmap='viridis')

        fig.colorbar(cax, ax=ax)

        ax.set_xticks(np.arange(embeddings_dimension))
        ax.set_yticks(np.arange(tokenized_input_length))
        ax.set_xticklabels([f'Dim {i}' for i in range(embeddings_dimension)])
        ax.set_yticklabels([f'Token {i}' for i in range(tokenized_input_length)])

        plt.xticks(rotation=90)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        for i in range(tokenized_input_length):
            for j in range(embeddings_dimension):
                text = ax.text(j, i, f'{embeddings_np[batch_idx, i, j]:.2f}', ha='center', va='center', color='white')

        ax.set_title(f"{title} - Batch {batch_idx + 1}")

    plt.show()


def createLossPlot(steps, losses, title="training"):
    plt.plot(steps, losses, linewidth=1)
    plt.xlabel("steps")
    plt.ylabel("losses")
    plt.title(title)
    plt.show()


class LlamaVisualizer:
    def __init__(self, model):
        self.model = model
        self.graph = nx.DiGraph()
        self.extract_model_details()

    def extract_model_details(self):
        self.layers = []
        for i, layer in enumerate(self.model.layers):
            self.layers.append({
                'self_attn': layer.attention,
                'ffn': layer.feed_forward
            })

    def build_graph(self):
        # Add nodes for each layer, neuron, weights, and biases
        self.graph.add_node("Input", layer=0)
        self.graph.add_node("TokenEmbeddings", layer=1)

        for i, layer in enumerate(self.layers):
            attn_node = f"Layer{i+1}_SelfAttention"
            ffn_node = f"Layer{i+1}_FeedForward"

            self.graph.add_node(attn_node, layer=i+2, weight=self.get_layer_weight(layer['self_attn']))
            self.graph.add_node(ffn_node, layer=i+2, weight=self.get_layer_weight(layer['ffn']))

            self.graph.add_edge("TokenEmbeddings", attn_node, weight=self.get_layer_weight(layer['self_attn']))
            self.graph.add_edge(attn_node, ffn_node, weight=self.get_layer_weight(layer['ffn']))
            self.graph.add_edge(ffn_node, "Output", weight=None)

        self.graph.add_node("Output", layer=len(self.layers) + 2)

    def get_layer_weight(self, layer):
        if isinstance(layer, nn.Module):
            # Summarize the weights for visualization purposes
            return sum(p.numel() for p in layer.parameters())
        return None

    def visualize(self):
        self.build_graph()

        pos = nx.multipartite_layout(self.graph, subset_key="layer")
        plt.figure(figsize=(20, 10))

        # Draw the nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=7000, node_color="skyblue")

        # Draw the edges with weights
        edges = self.graph.edges(data=True)
        nx.draw_networkx_edges(self.graph, pos, edgelist=edges, arrowstyle='->', arrowsize=20, edge_color="gray")

        # Draw the labels
        labels = {n: f"{n}\n(layer={d['layer']})\n(weight={d.get('weight', 'N/A')})" for n, d in self.graph.nodes(data=True)}
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=8)

        plt.title("Simplified Llama Model Architecture")
        plt.show()


def plot_lm_head_output(output_features=111000, title="LM Head Output Features"):
    output_tensor_np = output_features.numpy()

    # Plot the output tensor for the first batch
    plt.figure(dpi=500, figsize=(10, 6))
    plt.imshow(output_tensor_np[0], aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Output Features')
    plt.ylabel('Sequence Length')
    plt.show()

def plot_probs_or_logits(probs, title="Probabilities", ylabel="Probability", xlabel="Output Tokens", label="Probabilities"):
    probs_np = probs.numpy() if isinstance(probs, torch.Tensor) else probs

    plt.figure(dpi=500, figsize=(10, 6))
    plt.plot(probs_np[0], label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_mask_tensor(mask, title="Mask"):
    mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask

    plt.figure(dpi=500, figsize=(10, 6))
    plt.imshow(mask_np, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Position")
    plt.ylabel("Position")
    plt.show()

def plot_intermediate_attention(scores, title="Attention Scores", xlabel="Keys", ylabel="Queries"):
    scores_np = scores.clone().detach().cpu().numpy()
    
    plt.figure(dpi=500, figsize=(10, 6))
    plt.imshow(scores_np[0][0], aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()