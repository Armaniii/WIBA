# Install required packages if not already installed
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("httr")) install.packages("httr")
if (!require("jsonlite")) install.packages("jsonlite")

# Load required libraries
library(tidyverse)
library(httr)
library(jsonlite)

# Base URL for WIBA API
BASE_URL <- "http://wiba.dev/api"

# Function to create segments
create_segments <- function(input_data, column_name, window_size = 3) {
  url <- paste0(BASE_URL, "/create_segments")
  
  # Convert data frame to CSV string
  csv_data <- capture.output(write.csv(input_data, row.names = FALSE))
  csv_string <- paste(csv_data, collapse = "\n")
  
  payload <- list(
    data = csv_string,
    column_name = column_name,
    window_size = window_size,
    step_size = 1
  )
  
  response <- POST(
    url,
    body = payload,
    encode = "json"
  )
  
  if (status_code(response) == 200) {
    result <- fromJSON(rawToChar(response$content))
    segments_df <- as.data.frame(result)
    print("Created segments:")
    print(segments_df)
    return(segments_df)
  } else {
    stop(paste("Error:", status_code(response), content(response)))
  }
}

# Function to detect arguments
wiba_detect <- function(df) {
  url <- paste0(BASE_URL, "/detect")
  
  # Convert data frame to CSV string
  csv_string <- capture.output(write.csv(df, row.names = FALSE))
  csv_string <- paste(csv_string, collapse = "\n")
  
  payload <- list(texts = csv_string)
  
  response <- POST(
    url,
    body = payload,
    encode = "json"
  )
  
  if (status_code(response) == 200) {
    result <- fromJSON(rawToChar(response$content))
    return(as.data.frame(result))
  } else {
    stop(paste("Error:", status_code(response), content(response)))
  }
}

# Function to extract topics
wiba_extract <- function(texts) {
  url <- paste0(BASE_URL, "/extract")
  
  payload <- list(texts = texts)
  
  response <- POST(
    url,
    body = payload,
    encode = "json"
  )
  
  if (status_code(response) == 200) {
    result <- fromJSON(rawToChar(response$content))
    return(as.data.frame(result))
  } else {
    stop(paste("Error:", status_code(response), content(response)))
  }
}

# Function to analyze stance
wiba_stance <- function(texts, topics) {
  url <- paste0(BASE_URL, "/stance")
  
  payload <- list(
    texts = texts,
    topics = topics
  )
  
  response <- POST(
    url,
    body = payload,
    encode = "json"
  )
  
  if (status_code(response) == 200) {
    result <- fromJSON(rawToChar(response$content))
    return(as.data.frame(result))
  } else {
    stop(paste("Error:", status_code(response), content(response)))
  }
}

# Create example dataset
create_example_dataset <- function() {
  data <- list(
    id = as.character(1:21),
    parent_id = c(NA, rep("1", 3), NA, rep("5", 2), NA, rep("8", 2), NA, 
                 rep("11", 2), NA, rep("14", 2), NA, rep("17", 3), "20"),
    thread_id = c(rep("1", 4), rep("5", 3), rep("8", 3), rep("11", 3),
                 rep("14", 3), rep("17", 5)),
    text = c(
      # Thread 1
      "I strongly believe that this new policy will significantly reduce pollution...",
      "While the goal of reducing pollution is certainly commendable...",
      "I understand the concerns about economic costs...",
      "I completely agree with you on the long-term perspective...",
      
      # Thread 2
      "The new law aimed at improving education outcomes...",
      "While I agree that improving education is essential...",
      "The city's new traffic management plan has sparked...",
      
      # Thread 3
      "The city's traffic management plan is flawed...",
      "While the bike lanes cause issues...",
      "We need better planning and enforcement...",
      
      # Additional arguments
      "Public transportation should be improved...",
      "Subsidizing small businesses to comply...",
      "Teacher training reforms will have...",
      "I don't think the policy will reduce emissions...",
      "Banning cars from the city center...",
      "Encouraging carpooling would also reduce traffic...",
      "The education reform law should focus...",
      "Funding for renewable energy should be prioritized...",
      "Tax incentives for green technologies...",
      "Road pricing would better address congestion...",
      "More public awareness campaigns are needed..."
    )
  )
  
  return(as.data.frame(data))
}

# Example usage
main <- function() {
  # Create example dataset
  df <- create_example_dataset()
  print("Created example dataset:")
  print(head(df))
  
  # Create segments
  segments_df <- create_segments(df, "text", window_size = 3)
  
  # Detect arguments
  detect_df <- wiba_detect(df)
  print("\nDetected arguments:")
  print(head(detect_df))
  
  # Extract topics
  texts <- detect_df$text
  extract_df <- wiba_extract(texts)
  print("\nExtracted topics:")
  print(head(extract_df))
  
  # Analyze stance
  topics <- extract_df$topic
  stance_df <- wiba_stance(texts, topics)
  print("\nStance analysis:")
  print(head(stance_df))
  
  # Combine results
  result_df <- bind_cols(detect_df, extract_df, stance_df)
  print("\nCombined results:")
  print(head(result_df))
  
  # Return all results
  return(list(
    original_data = df,
    segments = segments_df,
    detection = detect_df,
    topics = extract_df,
    stance = stance_df,
    combined = result_df
  ))
}

# Example of running the analysis
results <- main()
