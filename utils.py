import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import re, os

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
    plt.figure(dpi=600, figsize=(60, 6))
    plt.imshow(output_tensor_np[0], aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Output Features')
    plt.ylabel('Sequence Length')
    plt.show()

def plot_probs_or_logits(probs, title="Probabilities", ylabel="Probability", xlabel="Output Tokens", label="Probabilities"):
    probs_np = probs.detach().numpy()

    plt.figure(dpi=500, figsize=(10, 6))
    plt.plot(probs_np[0], label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_mask_tensor(mask, title="Mask"):
    mask_np = mask.detach().numpy()

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

def parse_parameters_from_file(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        content = file.read()
    parameter_blocks = content.strip().split('\n\n')
    for block in parameter_blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue
        name_line = lines[0]
        data_lines = lines[1:]
        name_match = re.match(r"Name: (.+), Shape", name_line)
        if not name_match:
            continue
        name = name_match.group(1)
        data_str = ''.join(data_lines)
        data = torch.tensor(eval(data_str))
        parameters[name] = data
    return parameters

def visualize_parameters(parameters, combined_file_path, individual_folder_path):
    n = len(parameters)
    cols = 4  # Number of columns in the plot grid
    rows = (n + cols - 1) // cols  # Calculate number of rows needed

    if not os.path.exists(individual_folder_path):
        os.makedirs(individual_folder_path)

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5), dpi=600)

    for ax, (name, param) in zip(axes.flatten(), parameters.items()):
        fig_individual, ax_individual = plt.subplots()
        if param.dim() == 1:
            ax.plot(param.numpy())
            ax.set_title(f'{name} (Bias)')
            ax_individual.plot(param.numpy())
            ax_individual.set_title(f'{name} (Bias)')
        elif param.dim() == 2:
            im = ax.imshow(param.numpy(), aspect='auto', cmap='viridis')
            fig.colorbar(im, ax=ax)
            ax.set_title(f'{name} (Weights)')
            im_individual = ax_individual.imshow(param.numpy(), aspect='auto', cmap='viridis')
            fig_individual.colorbar(im_individual, ax=ax_individual)
            ax_individual.set_title(f'{name} (Weights)')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax_individual.set_xlabel('Dimension 1')
        ax_individual.set_ylabel('Dimension 2')
        
        # Save individual plot
        individual_file_path = os.path.join(individual_folder_path, f"{name.replace('.', '_')}.png")
        fig_individual.savefig(individual_file_path)
        plt.close(fig_individual)

    # Remove empty subplots
    for ax in axes.flatten()[n:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig(combined_file_path)
    plt.show()

def save_model_parameters_to_file(model, file_path):
    torch.set_printoptions(profile="full")
    with open(file_path, 'w') as file:
        for name, param in model.named_parameters():
            file.write(f"Name: {name}, Shape: {param.shape}\n")
            file.write(f"{param.data.tolist()}\n\n")
    torch.set_printoptions(profile="default")