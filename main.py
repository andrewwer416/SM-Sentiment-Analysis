# Authors: Andrew Wernersbach, Jacob Zaslav



import API_client
import data_processing
import train

def main():
    df_test = data_processing.read()
    [X_train, X_val, y_train, y_val] = data_processing.preprocess(df_test)
    train.train(X_train, X_val, y_train, y_val)


if __name__ == "__main__":
    main()
    

