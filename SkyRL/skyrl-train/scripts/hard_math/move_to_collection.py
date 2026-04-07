from huggingface_hub import create_collection, add_collection_item

# --- CONFIGURATION ---
COLLECTION_NAME = "RLVR with Noisy Data"
ORG_NAME = "uiuc-kang-lab"
COLLECTION_DESCRIPTION = ""
# List of full model IDs (namespace/repo_name)
MODEL_IDS = [
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-clean-epoch-3",
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-clean-epoch-4",
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-0.1-epoch-3",
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-0.2-epoch-3",
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-0.3-epoch-3",
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-0.4-epoch-3",
    "uiuc-kang-lab/Qwen2.5-Math-7B-PGFC-noise-0.5-epoch-3",
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-epoch-3",
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-format-epoch-3",
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-random-epoch-2",
    "uiuc-kang-lab/Qwen2.5-Math-7B-DAPO-noise-0.5-epoch-3",
    "uiuc-kang-lab/Qwen2.5-Math-7B-DrGRPO-noise-0.5-epoch-3",
    "uiuc-kang-lab/Qwen2.5-Math-7B-SAPO-noise-0.5-epoch-3",
    "uiuc-kang-lab/Qwen2.5-Math-7B-TIS-noise-0.5-epoch-3",
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-0.5-epoch-3"
]

def organize_models():
    try:
        # 1. Create the collection (or get the existing one)
        # Note: If it already exists, this might require fetching the slug
        print(f"Checking for collection: {COLLECTION_NAME}...")
        collection = create_collection(
            title=COLLECTION_NAME,
            description=COLLECTION_DESCRIPTION,
            exists_ok=True,
            namespace=ORG_NAME
        )
        
        collection_slug = collection.slug
        print(f"Using collection: {collection_slug}")

        # 2. Add each model to the collection
        for model_id in MODEL_IDS:
            print(f"Adding {model_id} to collection...")
            add_collection_item(
                collection_slug=collection_slug,
                item_id=model_id,
                item_type="model",
                exists_ok=True
            )
        
        print("\nSuccess! All models have been added.")
        print(f"View your collection at: https://huggingface.co/collections/{collection_slug}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    organize_models()