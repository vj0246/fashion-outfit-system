# Dare XAI – Machine Learning & AI Engineer Intern Assignment
## Assignment: AI Fashion Outfit Recommendation System

Welcome to the **Dare XAI Fashion Outfit Recommendation System** dataset! This package contains a curated subset of fashion items and pre-styled outfits designed to evaluate your capabilities in computer vision, recommendation systems, search retrieval, and natural language understanding.

---

## 📁 Dataset Directory Structure

This curated dataset is organized as follows:
```text
NEWDATASET/
├── README.md               # This documentation file
├── curated25.xlsx          # Original styled outfits spreadsheet
├── outfits.csv             # Cleaned outfit mapping of the 25 curated outfits
├── products.csv            # Detailed metadata for the 68 unique products used in the outfits
└── images/                 # Product image files matching the product IDs
    ├── ajio/               # Images sourced from Ajio
    ├── myntra/             # Images sourced from Myntra
    └── nykaa/              # Images sourced from Nykaa
```

---

## 📊 Data Files Description

### 1. `products.csv`
This file contains the core metadata for the 68 unique fashion items that make up our outfits.
* **Fields**:
  * `id`: Unique identifier for the product (e.g., `ajio_703182002`).
  * `name`: Product title (e.g., `Women Bodycon Midi Length Dress`).
  * `brand`: Manufacturer or label (e.g., `Fyre Rose`, `Peter England`).
  * `price_inr`: Retail price in Indian Rupees (INR).
  * `rating` / `rating_count`: Customer rating statistics.
  * `gender`: Target gender (`men` / `women`).
  * `wear_type`: Style category (e.g., `western`, `ethnic`).
  * `category` & `category_label`: Specific clothing/accessory category (e.g., `formal-shirts`, `heels`, `dresses`).
  * `occasion`: Intended setting (e.g., `party`, `office`, `casual`).
  * `tags`: Semicolon-separated tags for retrieval.
  * `description`: Detailed text description of the product.
  * `image`: Relative filepath to the product image (e.g., `images/ajio/703182002.jpg`).

### 2. `outfits.csv` (and `curated25.xlsx`)
This file defines 25 expert-curated complete outfits. You can use this file as ground truth for training, evaluation, or as reference combinations.
* **Fields**:
  * `outfit_id`: Unique identifier for the outfit (e.g., `outfit W1`).
  * `gender` / `wear_type` / `occasion` / `theme`: Categorization context.
  * `hero` & `hero_id`: The main item in the outfit (e.g., a dress or shirt).
  * `second` & `second_id`: The complementary item (e.g., trousers/chinos).
  * `layer` & `layer_id`: Optional layering item (e.g., blazers, jackets).
  * `footwear` & `footwear_id`: Footwear item.
  * `accessory_1` & `accessory_1_id` / `accessory_2` & `accessory_2_id`: Optional styling accessories.
  * `palette`: Main color combination.
  * `stylist_rationale`: Stylist commentary explaining why this outfit is compatible and fits the theme.

---

## 🎯 Assignment & Problem Statement

Your objective is to design and build an intelligent **Recommendation Engine & Chat-based Fashion Assistant** that can understand natural language user requests, retrieve compatible clothing items, compile complete outfits, and explain its reasoning.

### Core Implementation Requirements:

1. **Dataset Analysis & Understanding**:
   * Inspect the provided metadata and images.
   * Document categories, palette distributions, and potential challenges (e.g., size of dataset, variety, metadata consistency).

2. **Outfit Compatibility Engine**:
   * Build an algorithm to determine if items are compatible. Given a single item (e.g., a white formal shirt), the engine should suggest compatible components (e.g., navy trousers and brown loafers).
   * **Tip**: Use similarity search or learn a pairwise compatibility score using visual/text features.

3. **User & Context-Aware Recommendations**:
   * Adapt recommendations dynamically based on profile parameters:
     * **Gender** (e.g., Men / Women)
     * **Age Group** (e.g., 20s vs. 40s styling)
     * **Occasion** (e.g., Office, Beach Vacation, Wedding, Party)
     * **Style Preferences** (e.g., Formal, Smart Casual)

4. **Conversational Fashion Assistant (Natural Language Interface)**:
   * Build a chat interface allowing users to make requests in plain text (e.g., *"I need an outfit for a business meeting"* or *"Suggest something stylish for a summer beach vacation"*).
   * The assistant should retrieve the items, group them into a complete outfit (Topwear, Bottomwear, Footwear, and optional Accessories/Layers), and display them to the user.

5. **Explainability**:
   * Every outfit recommendation must include a reasoned explanation (e.g., *"Beige chinos pair well with a navy blazer because they provide classic contrast while maintaining a polished smart-casual appearance for your office meeting."*).

---

## 🛠️ Recommended Technical Approach

We evaluate technical depth and systemic engineering choices. Consider incorporating:
* **Computer Vision & Multi-modal Embeddings**: Use models like **CLIP**, **FashionCLIP**, or **SigLIP** to generate embeddings from both product images and descriptions.
* **Vector Search / Retrieval**: Store product embeddings in a vector database (e.g., **Qdrant**, **Chroma**, **FAISS**) to execute fast similarity and hybrid searches.
* **LLM Integration**: Use LLMs (e.g., Gemini, GPT, Claude) to parse user intent from conversational chat, structure queries, and generate final personalized reasoning.
* **Advanced Methods (Bonus)**: Graph-based recommendations (representing outfits as nodes/edges) or trained compatibility classification models.

---

## 🚀 Quick Start Code (Python)

You can load and start exploring this dataset using the following snippet:

```python
import pandas as pd
import os

# Set paths
DATASET_DIR = "./"  # Update path if run from elsewhere
products_df = pd.read_csv(os.path.join(DATASET_DIR, "products.csv"))
outfits_df = pd.read_csv(os.path.join(DATASET_DIR, "outfits.csv"))

print(f"Loaded {len(products_df)} products.")
print(f"Loaded {len(outfits_df)} curated outfits.")

# Example: Display first outfit
first_outfit = outfits_df.iloc[0]
print(f"\nOutfit ID: {first_outfit['outfit_id']} ({first_outfit['theme']})")
print(f"Hero Item: {first_outfit['hero']} (ID: {first_outfit['hero_id']})")
print(f"Footwear: {first_outfit['footwear']} (ID: {first_outfit['footwear_id']})")
print(f"Rationale: {first_outfit['stylist_rationale']}")
```

---
*Good luck with the assignment! We look forward to seeing your creative and technical solutions.*
# ML-Intern-Task
