import requests
import pandas as pd

# Base URL of the API
BASE_URL = 'http://wiba.dev/api'

def wiba_extract(texts):
    """
    Extract information from given texts using the API's extract endpoint.
    
    Args:
        texts (list): List of text strings to analyze.
    
    Returns:
        pandas.DataFrame: Results of the extraction.
    """
    url = f"{BASE_URL}/extract"
    payload = {"texts": texts}
    response = requests.post(url, json=payload)
    return pd.DataFrame(response.json())

def wiba_detect(df):
    """
    Detect information in texts from a DataFrame using the API's detect endpoint.
    
    Args:
        df (pandas.DataFrame): DataFrame containing texts to analyze.
    
    Returns:
        pandas.DataFrame: Results of the detection.
    """
    url = f"{BASE_URL}/detect"
    csv_data = df.to_csv(index=False)
    payload = {"texts": csv_data}
    response = requests.post(url, json=payload)
    return pd.DataFrame(response.json())

def wiba_stance(texts, topics):
    """
    Analyze the stance of texts on given topics using the API's stance endpoint.
    
    Args:
        texts (list): List of text strings to analyze.
        topics (list): List of topics to analyze stance on.
    
    Returns:
        pandas.DataFrame: Results of the stance analysis.
    """
    url = f"{BASE_URL}/stance"
    payload = {"texts": texts, "topics": topics}
    response = requests.post(url, json=payload)
    return pd.DataFrame(response.json())

def create_segments(input_file, column_name, window_size=3):
    """
    Create text segments using the API's create_segments endpoint.
    
    Args:
        input_file (str): CSV file content as a string.
        column_name (str): Name of the column containing text to segment.
        window_size (int, optional): Size of the sliding window. Defaults to 3.
    
    Returns:
        pandas.DataFrame: Created segments.
    """
    url = f"{BASE_URL}/create_segments"
    payload = {
        "data": input_file,
        "column_name": column_name,
        "window_size": window_size,
        "step_size": 1
    }
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result_csv = response.json()
        segments_df = pd.DataFrame(result_csv)
        print("Created segments:")
        print(segments_df)
        return segments_df
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def calculate_segments(input_file, column_name, confidence_column):
    """
    Calculate segment scores using the API's calculate_segments endpoint.
    
    Args:
        input_file (str): CSV file content as a string.
        column_name (str): Name of the column containing segments.
        confidence_column (str): Name of the column containing confidence scores.
    
    Returns:
        pandas.DataFrame: Calculated segment scores.
    """
    url = f"{BASE_URL}/calculate_segments"
    payload = {
        "data": input_file,
        "segment_column": column_name,
        "argument_score_column": confidence_column
    }
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result_csv = response.json()
        segments_df = pd.DataFrame(result_csv)
        print("Calculated segments:")
        return segments_df
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Example usage
if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv("PATH/TO/YOUR/DATA")
    print("Sample data:")
    print(df.head())
    create_segments_df = create_segments(df_csv, "text", 3)

    if create_segments_df is not None:
        # Detect texts
        detect_df = wiba_detect(create_segments_df)
        print("Detection results:")
        print(detect_df)

        # Extract topics from texts
        texts = detect_df['text'].tolist()
        extract_df = wiba_extract(texts)
        print("Extraction results:")
        print(extract_df)

        # Analyze stance on topics
        topics = extract_df['topic'].tolist()
        stance_df = wiba_stance(texts, topics)
        print("Stance results:")
        print(stance_df)

        # Calculate segments
        detect_csv = detect_df.to_csv(index=False)
        segments = calculate_segments(detect_csv, 'text', 'argument_confidence')
        if segments is not None:
            print("Calculated segments:")
            print(segments)

