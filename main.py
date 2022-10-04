# Authors: Andrew Wernersbach, Jacob Zaslav



import API_client
import data_processing

def main():
    df_test = data_processing.read()
    data_processing.preprocess(df_test)


if __name__ == "__main__":
    main()

