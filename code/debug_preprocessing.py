from DEBUG_preprocessing_01 import load_data

data = load_data()

train_loader = data["train_loader"]

batch = next(iter(train_loader))

print("\n[DEBUG SCRIPT] First train batch")
print("descriptor:", batch["descriptor"].shape)
print("covariate:", batch["covariate"].shape)
print("target:", batch["target"].shape)
print("mask:", batch["mask"].shape)
print("participant_id:", batch["participant_id"][:10])