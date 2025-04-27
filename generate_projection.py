import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate a random projection matrix and save it.")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="Hidden dimension of the LLM (d)")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension (k)")
    parser.add_argument("--output-file", type=str, default="Wc.bin", help="Output file path for the matrix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"Generating projection matrix Wc with shape ({args.hidden_dim}, {args.latent_dim})")
    np.random.seed(args.seed)
    # Standard normal initialization - good default
    Wc = np.random.randn(args.hidden_dim, args.latent_dim).astype(np.float32)

    print(f"Saving matrix to {args.output_file}")
    Wc.tofile(args.output_file)

    print("Done.")

if __name__ == "__main__":
    main() 