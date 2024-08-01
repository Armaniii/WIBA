library(httr)
library(jsonlite)
library(dplyr)
library(readr)

# Base URL of the API
BASE_URL <- 'http://wiba.dev/api'

wiba_extract <- function(texts) {
  # Extract information from given texts using the API's extract endpoint
  url <- paste0(BASE_URL, "/extract")
  payload <- list(texts = texts)
  response <- POST(url, body = payload, encode = "json")
  content <- content(response, "text", encoding = "UTF-8")
  return(fromJSON(content))
}

wiba_detect <- function(df) {
  # Detect information in texts from a dataframe using the API's detect endpoint
  url <- paste0(BASE_URL, "/detect")
  csv_data <- write.csv(df, file = "", row.names = FALSE)
  payload <- list(texts = csv_data)
  response <- POST(url, body = payload, encode = "json")
  content <- content(response, "text", encoding = "UTF-8")
  return(fromJSON(content))
}

wiba_stance <- function(texts, topics) {
  # Analyze the stance of texts on given topics using the API's stance endpoint
  url <- paste0(BASE_URL, "/stance")
  payload <- list(texts = texts, topics = topics)
  response <- POST(url, body = payload, encode = "json")
  content <- content(response, "text", encoding = "UTF-8")
  return(fromJSON(content))
}

create_segments <- function(input_file, column_name, window_size = 3) {
  # Create text segments using the API's create_segments endpoint
  url <- paste0(BASE_URL, "/create_segments")
  payload <- list(
    data = input_file,
    column_name = column_name,
    window_size = window_size,
    step_size = 1
  )
  response <- POST(url, body = payload, encode = "json")
  
  if (status_code(response) == 200) {
    content <- content(response, "text", encoding = "UTF-8")
    segments_df <- fromJSON(content)
    print("Created segments:")
    print(segments_df)
    return(segments_df)
  } else {
    print(paste("Error:", status_code(response), content(response, "text")))
    return(NULL)
  }
}

calculate_segments <- function(input_file, column_name, confidence_column) {
  # Calculate segment scores using the API's calculate_segments endpoint
  url <- paste0(BASE_URL, "/calculate_segments")
  payload <- list(
    data = input_file,
    segment_column = column_name,
    argument_score_column = confidence_column
  )
  response <- POST(url, body = payload, encode = "json")
  
  if (status_code(response) == 200) {
    content <- content(response, "text", encoding = "UTF-8")
    segments_df <- fromJSON(content)
    print("Calculated segments:")
    return(segments_df)
  } else {
    print(paste("Error:", status_code(response), content(response, "text")))
    return(NULL)
  }
}

# Example usage
main <- function() {
  # Load sample data
  df <- read_csv("sample_data.csv")
  print("Sample data:")
  print(head(df))

  # Create segments
  df_csv <- write.csv(df, file = "", row.names = FALSE)
  create_segments_df <- create_segments(df_csv, "text", 3)

  if (!is.null(create_segments_df)) {
    # Detect texts
    detect_df <- wiba_detect(create_segments_df)
    print("Detection results:")
    print(detect_df)

    # Extract topics from texts
    texts <- detect_df$text
    extract_df <- wiba_extract(texts)
    print("Extraction results:")
    print(extract_df)

    # Analyze stance on topics
    topics <- extract_df$topic
    stance_df <- wiba_stance(texts, topics)
    print("Stance results:")
    print(stance_df)

    # Calculate segments
    detect_csv <- write.csv(detect_df, file = "", row.names = FALSE)
    segments <- calculate_segments(detect_csv, 'text', 'argument_confidence')
    if (!is.null(segments)) {
      print("Calculated segments:")
      print(segments)
    }
  }
}

# Run the main function
main()
