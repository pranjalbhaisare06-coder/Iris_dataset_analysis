import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

def train_and_save_model(data_path='Iris.csv', model_filename='iris_knn_model.pkl'):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Drop the 'Id' column
    df = df.drop(columns=['Id'])

    # Encode the 'Species' column
    le = LabelEncoder()
    df['Species'] = le.fit_transform(df['Species'])

    # Separate features (X) and target (y)
    X = df.drop(columns=['Species'])
    y = df['Species']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Using random_state for reproducibility

    # Initialize and train the KNeighborsClassifier model
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model (optional, for verification)
    accuracy = model.score(X_test, y_test) * 100
    print(f"Trained KNeighborsClassifier Accuracy: {accuracy:.2f}%")

    # Save the trained model
    joblib.dump(model, model_filename)
    print(f'KNeighborsClassifier model saved as {model_filename}')

if __name__ == '__main__':
    # Assuming Iris.csv is in the same directory as this script
    # or provide the correct path to the CSV file
    train_and_save_model(data_path='/content/Iris.csv')
