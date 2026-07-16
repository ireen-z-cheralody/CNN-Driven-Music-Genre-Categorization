import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import json
import os

from config import GENRES  # ['blues','classical','country','hiphop','jazz','pop','rock']

DATA_PATH = "data/gtzan/features_3_sec.csv"
RESULTS_PATH = "results/baseline_svm_results.json"


def main():
    df = pd.read_csv(DATA_PATH)
    print("Loaded full dataset:", df.shape)

    # keep only the 7 genres used by the CNN, so the comparison is apples-to-apples
    df = df[df["label"].isin(GENRES)].reset_index(drop=True)
    print("After filtering to 7 genres:", df.shape)

    # song id = original track name, e.g. "blues.00000" from "blues.00000.0.wav"
    df["song_id"] = df["filename"].str.extract(r"(^[a-z]+\.\d+)")

    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X = df.drop(columns=["filename", "label", "length", "song_id"])

    # split by SONG, not by segment, so clips from the same track never
    # appear in both train and test (avoids inflated accuracy from leakage)
    songs = df["song_id"].unique()
    train_songs, test_songs = train_test_split(songs, test_size=0.2, random_state=42)

    train_mask = df["song_id"].isin(train_songs)
    test_mask = df["song_id"].isin(test_songs)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = SVC(kernel="rbf", C=10, gamma="scale")
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    print(f"\nBaseline SVM test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    os.makedirs("results", exist_ok=True)
    results = {
        "model": "SVM (RBF kernel)",
        "genres": le.classes_.tolist(),
        "train_songs": len(train_songs),
        "test_songs": len(test_songs),
        "train_segments": int(train_mask.sum()),
        "test_segments": int(test_mask.sum()),
        "test_accuracy": acc,
        "classification_report": report,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to", RESULTS_PATH)


if __name__ == "__main__":
    main()