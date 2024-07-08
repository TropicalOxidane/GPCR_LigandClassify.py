import time
import random
import sys
from pathlib import Path
import seaborn as sns

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import Draw, rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
from openTSNE import TSNE
import matplotlib as plt
from matplotlib import axis
from mpl_toolkits.mplot3d import Axes3D
from rdkit.Chem.Draw import SimilarityMaps

"""Adopted protocol for ligand/receptor filtering,
    data cleaning and pre-preprocessing
"""


# Define functions for merging and processing TSV files
def merge_tsv_files(ligands, interactions_active, targets, output_file):
    df1 = pd.read_csv(ligands, sep="\t")
    df2 = pd.read_csv(interactions_active, sep="\t")
    df3 = pd.read_csv(targets, sep="\t")

    merged_df = pd.concat([df1, df2, df3], axis=1)
    merged_df.to_csv(output_file, sep="\t", index=False)
    print(f"Merged TSV files and saved to {output_file}")


def process_tsv_file(file_path, gpcr_column_name, output_file):
    df = pd.read_csv(file_path, sep="\t")

    def clean_column_headers(df):
        df.columns = df.columns.str.replace(
            r"[^\w\s]", "", regex=True
        )  # Remove special characters
        df.columns = df.columns.str.replace(" ", "")  # Remove spaces
        return df

    def group_gpcr_subtypes(df, gpcr_column):
        subtype_mapping = {
            "Alpha-1A adrenergic receptor": "Alpha-adrenergic receptor",
            "Alpha-1B adrenergic receptor": "Alpha-adrenergic receptor",
            "Alpha-1D adrenergic receptor": "Alpha-adrenergic receptor",
            "Alpha-2A adrenergic receptor": "Alpha-adrenergic receptor",
            "Alpha-2B adrenergic receptor": "Alpha-adrenergic receptor",
            "Alpha-2C adrenergic receptor": "Alpha-adrenergic receptor",
        }
        if gpcr_column in df.columns:
            df[gpcr_column] = df[gpcr_column].replace(subtype_mapping)
        return df

    df = clean_column_headers(df)

    if gpcr_column_name not in df.columns:
        raise KeyError(
            f"Column '{gpcr_column_name}' not found in {file_path}. Available columns: {df.columns.tolist()}"
        )

    df = group_gpcr_subtypes(df, gpcr_column_name)

    if "MolecularWeight" in df.columns and "XlogP" in df.columns:
        df = df[(df["MolecularWeight"] >= 100) & (df["MolecularWeight"] <= 900)]
        df = df[(df["XlogP"] >= -4) & (df["XlogP"] <= 10)]
    else:
        print(
            "Warning: 'MolecularWeight' and/or 'XlogP' columns not found. Skipping this filter."
        )

    df.to_csv(output_file, sep="\t", index=False)
    print(f"Processed TSV file and saved the result to {output_file}")
    return df


# Function to add an empty activity column
def add_empty_activity_column(file_path, output_file):
    df = pd.read_csv(file_path, sep="\t")
    df["Activity"] = ""  # Add an empty column for activity
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Added empty activity column and saved the result to {output_file}")
    return df


# Paths to input files
ligands = r"C:\\Users\\gavjo\\OneDrive\\Documents\\TSV_2\\ligands.tsv"
interactions_active = (
    r"C:\\Users\\gavjo\\OneDrive\\Documents\\TSV_2\\interactions_active.tsv"
)
targets = r"C:\\Users\\gavjo\\OneDrive\\Documents\\TSV_2\\targets.tsv"
merged_file = "merged.tsv"
processed_file = "filtered_merged.tsv"
output_file_with_activity = "filtered_merged_with_activity.tsv"
gpcr_column_name = "GPCRName"

# Merge and process files
merge_tsv_files(ligands, interactions_active, targets, merged_file)
final_df = process_tsv_file(merged_file, gpcr_column_name, processed_file)
print(f"Processed TSV file and saved the result to {processed_file}")

# Add empty activity column
final_df_with_activity = add_empty_activity_column(
    processed_file, output_file_with_activity
)
print(
    f"Added empty activity column and saved the result to {output_file_with_activity}"
)
print(final_df_with_activity.head())

"""Molecular featurization using the DeepChem library
"""


# Function to calculate molecular fingerprints
def calculate_fingerprints(smiles):
    global features, ecfp4
    mol = Chem.MolFromSmiles(smiles)
    ecfp4 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=32)
    ecfp4_dict = {f"ECFP4_{i}": bit for i, bit in enumerate(ecfp4)}
    features = ecfp4
    return ecfp4_dict


# Load processed data
data = pd.read_csv(output_file_with_activity, sep="\t")

# Extract descriptors for each SMILES
descriptor_list = []
for smiles in data["SMILES"]:
    descriptor_list.append(calculate_fingerprints(smiles))

descriptor_df = pd.DataFrame(descriptor_list)
X = descriptor_df

# Convert GPCR column to a list of GPCR targets for each ligand
y = data[gpcr_column_name].apply(lambda x: x.split(";") if pd.notna(x) else [])

# Encode the multi-label targets
mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(y)

# Shuffle the entire DataFrame
shuffled_data = (
    pd.concat([X, pd.DataFrame(y_encoded, columns=mlb.classes_)], axis=1)
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    shuffled_data.drop(columns=mlb.classes_),
    shuffled_data[mlb.classes_],
    test_size=0.3,
    random_state=42,
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""Training and validation of the machine learning models
"""

"""Model’s training
"""

# Define the MLPClassifier with MultiOutputClassifier
mlp = MLPClassifier(verbose=1)
multi_target_mlp = MultiOutputClassifier(mlp)

# Define the grid search parameters
param_grid = {
    "estimator__hidden_layer_sizes": [(64,), (128,)],
    "estimator__learning_rate_init": [0.001, 0.01],
    "estimator__batch_size": [32, 64],
    "estimator__max_iter": [200, 300],
}

"""Optimization of models’ hyperparameters
"""

# Create Grid Search
grid = GridSearchCV(estimator=multi_target_mlp, param_grid=param_grid, n_jobs=-1, cv=2)
grid_result = grid.fit(X_train_scaled, y_train)

# Summarize results
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

# Evaluate the best model
best_model = grid_result.best_estimator_
y_pred = best_model.predict(X_test_scaled)

"""Model validation and assessment
"""

# Calculate the accuracy for each target
accuracies = [
    accuracy_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(y_test.shape[1])
]
print(f"Accuracies: {accuracies}")

# Save the best model
joblib.dump(best_model, "best_multi_target_gpcr_model.pkl")


# Function to predict GPCR targets of ligands in the dataset
def predict_targets(data, model, scaler, mlb, top_n=3):
    predictions = []
    for idx, smiles in enumerate(data["SMILES"]):
        descriptors = calculate_fingerprints(smiles)
        descriptor_df = pd.DataFrame([descriptors])
        descriptor_scaled = scaler.transform(descriptor_df)
        prediction_probs = model.predict_proba(descriptor_scaled)[0]
        top_indices = np.argsort(prediction_probs)[-top_n:][::-1]
        predicted_classes = mlb.classes_[top_indices]
        predictions.append(";".join(predicted_classes[0]))
        if idx % 100 == 0:
            print(f"Processed {idx+1}/{len(data)} rows")
    data["Predicted_GPCR"] = predictions
    return data


# Load the best model and scaler
loaded_model = joblib.load("best_multi_target_gpcr_model.pkl")

# Predict GPCR targets for the dataset
data_with_predictions = predict_targets(data, loaded_model, scaler, mlb)
output_prediction_file = "ligands_with_predicted_gpcr.tsv"
data_with_predictions.to_csv(output_prediction_file, sep="\t", index=False)
print(f"Predicted GPCR targets and saved the result to {output_prediction_file}")

# Combine the training and testing data
combined_data = np.vstack((X_train, X_test))

# tSNE - high-dimensional visualization
tsne = TSNE(n_components=3, random_state=42)
tsne_results = tsne.fit_transform(combined_data)

# Split the tSNE results back into training and testing data
tsne_train = tsne_results[: len(X_train)]
tsne_test = tsne_results[len(X_train) :]

# Plotting the results in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    tsne_test[:, 0],
    tsne_test[:, 1],
    tsne_test[:, 2],
    subset="blue",
    label="test",
    edgecolor="k",
    alpha=0.7,
)
ax.scatter(
    tsne_train[:, 0],
    tsne_train[:, 1],
    tsne_train[:, 2],
    subset="red",
    label="train",
    edgecolor="k",
    alpha=0.7,
)
ax.set_title("3D tSNE Visualization of Molecular Fingerprints")
ax.axes.set_xlim3d(left=-30, right=30)
ax.axes.set_ylim3d(left=-30, right=30)
ax.axes.set_zlim3d(left=-30, right=30)
ax.set_xlabel("tSNE-1")
ax.set_ylabel("tSNE-2")
ax.set_zlabel("tSNE-3")
ax.legend(loc="best")
plt.show()

# Columns set one
Columns_set_one = ["XlogP", "MolecularWeight"]

# loading dataset
df_for_set_one = pd.read_csv(
    output_prediction_file, sep="\t", skipinitialspace=True, usecols=Columns_set_one
)
data = sns.load_dataset(df_for_set_one)

# draw jointplot with
# kde kind
sns.jointplot(
    x=df_for_set_one.XlogP, y=df_for_set_one.MolecularWeight, kind="kde", data=data
)

# Show the plot
plt.show()

# Columns set two
Columns_set_two = ["SMILES", "LigandName"]

# loading dataset
df_for_set_two = pd.read_csv(
    output_prediction_file,
    sep="\t",
    skipinitialspace=True,
    usecols=Columns_set_two,
    index_col=0,
)
print(df_for_set_two.head())


# Credits to: @MunibaFaiza for the Tanimoto similarities code
# Creating molecules and storing in an array
molecules = []

for _, smiles in df_for_set_two[["SMILES"]].itertuples():
    molecules.append((Chem.MolFromSmiles(smiles)))
molecules[:15]

# Calculating fingerprints in molecules array
fgrps = [calculate_fingerprints.GetFingerprint(mol) for mol in molecules]

# Calculating number of fingerprints
nfgrps = len(fgrps)
print("Number of fingerprints:", nfgrps)


# Defining a function to calculate similarities among the molecules
def pairwise_similarity(fingerprints_list):

    global similarities

    similarities = np.zeros(nfgrps, nfgrps)

    for i in range(1, nfgrps):
        similarity = DataStructs.BulkTanimotoSimilarity(fgrps[i], fgrps[:i])
        similarities[i, :i] = similarity
        similarities[:i, i] = similarity

    return similarities


# Calculating similarities of molecules
pairwise_similarity(fgrps)
tri_lower_diag = np.tril(similarities, k=0)

# Visualizing the similarities
labels = [
    "lig1",
    "lig2",
    "lig3",
    "lig4",
    "lig5",
    "lig6",
    "lig7",
    "lig8",
    "lig9",
    "lig10",
    "lig11",
    "lig12",
    "lig13",
    "lig14",
    "lig15",
]


def normal_heatmap(sim):

    # writing similarities to a file
    f = open("similarities.txt", "w")
    print(similarities, file=f)

    sns.set(font_scale=0.8)

    # generating the plot

    plot = sns.heatmap(
        sim[:15, :15],
        annot=True,
        annot_kws={"fontsize": 5},
        center=0,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidth=0.7,
        cbar_kws={"shrink": 0.5},
    )

    plt.title("Heatmap of Tanimoto Similarities", fontsize=20)

    plt.show()

    # saving the plot

    fig = plot.get_figure()
    fig.savefig("tanimoto_heatmap.png")


normal_heatmap(similarities)


# Function to predict activity of new ligands
def predict_activity(smiles, model, scaler, label_encoder):
    descriptors = calculate_fingerprints(smiles)
    descriptor_df = pd.DataFrame([descriptors])
    descriptor_scaled = scaler.transform(descriptor_df)
    prediction = model.predict(descriptor_scaled)
    predicted_class = label_encoder.inverse_transform([prediction.argmax()])
    return predicted_class[0]


# Example usage
new_smiles = input("New smiles: ")
print("New smiles: " + new_smiles)
activity = predict_activity(new_smiles, scaler, mlb)
print(f"The predicted activity of the ligand is: {activity}")

output_file = r"C:\Users\gavjo\.spyder-py3\output_file.tsv"


def filter_tsv(output_file_with_activity, filter_column, filter_values):
    # Read the TSV file into a Dataframe
    df = pd.read_csv(output_file_with_activity, sep="\t")

    # Filter the Dataframe based on the given criteria
    filtered_df = df[df[filter_column].isin(filter_values)]

    # Write the filtered Datafrmae to a new TSV file
    filtered_df.to_csv(output_file, sep="\t", index=False)
    print(f"filtered data saved to {output_file}")


filter_column = "GPCRTarget"
filter_values = ["ADCY8", "ADTRP", "CACNA2D3", "CADPS2", "CDH20", "CDH8"]

# Print to output that the code's finished
print(f"finished: {activity}, {output_file}")
