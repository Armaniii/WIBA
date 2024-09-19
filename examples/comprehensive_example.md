# **Analyzing a Reddit-Style Dataset Using WIBA API and `delibtools`**

This notebook provides an end-to-end walkthrough for transforming a dataset of Reddit-style posts using the WIBA API for argument detection and stance analysis, followed by `delibtools` for calculating deliberation intensity.

### **1. Setup**

Start by installing the required libraries:

```bash
pip install requests pandas delibtools
```

### **2. Import Required Libraries**

```python
import requests
import pandas as pd
from delibtools import DeliberationIntensity
import json
```

### **3. Example Dataset**

We'll create a sample dataset containing 20 Reddit-style posts across three threads:

```python
data = {
    'id': [f'{i}' for i in range(1, 22)],
    'parent_id': [None, '1', '1', '1', None, '5', '5', None, '8', '8', None, '11', '11', None, '14', '14', None, '17', '17', '17', '20'],
    'thread_id': ['1', '1', '1', '1', '5', '5', '5', '8', '8', '8', '11', '11', '11', '14', '14', '14', '17', '17', '17', '17', '17'],
    'text': [
        # Thread 1
        "I strongly believe that this new policy will significantly reduce pollution in urban areas. Not only does it introduce more stringent regulations on industrial emissions, but it also promotes the adoption of green technologies in both public and private sectors. This is critical, as urban centers have long struggled with poor air quality due to vehicular and industrial emissions. By setting higher standards for emissions and offering incentives for green initiatives, the policy provides a comprehensive solution. It's not just about cutting emissions but also about fostering innovation in green tech. Additionally, the policy includes a long-term roadmap for monitoring and ensuring compliance, which is essential for its success over the coming decades.",
        "While the goal of reducing pollution is certainly commendable, I'm concerned that this policy might have unintended economic consequences. The costs associated with implementing these new regulations could be enormous, particularly for smaller businesses that may struggle to meet the new standards. In the short term, this could lead to increased costs for consumers, as businesses pass on the extra expenses. Moreover, there's a risk that some industries may relocate to regions with less stringent regulations, which would not only harm the local economy but also fail to achieve the policy's primary environmental goals. I'm also skeptical about the timeline for seeing meaningful improvements in air quality, given how complex and entrenched the problem is.",
        "I understand the concerns about economic costs, but I believe the long-term benefits far outweigh the short-term challenges. In the case of small businesses, the policy has provisions for subsidies and grants to help them transition to greener technologies. Moreover, we've seen from other case studies, such as the transition to renewable energy in certain parts of Europe, that once these systems are in place, they lead to cost savings and economic growth. Industries might indeed face challenges initially, but as green technologies become more widespread and affordable, the overall cost of compliance will decrease. I think it's important to focus on the bigger picture — cleaner air, improved public health, and long-term sustainability.",
        "I completely agree with you on the long-term perspective. It's easy to get caught up in the short-term costs, but we have to consider the health impacts of doing nothing. Chronic respiratory diseases, higher healthcare costs, and loss of productivity are all associated with high pollution levels, particularly in urban areas. This policy could lead to significant reductions in these health costs over time. Furthermore, investing in green technologies now could give our economy a competitive edge in the future global market, where sustainability is becoming an increasingly important factor. The policy isn’t perfect, but it’s a strong step in the right direction, provided the government stays committed to supporting industries during the transition.",
        
        # Thread 2
        "The new law aimed at improving education outcomes is a much-needed reform in our current system. By allocating more resources to underfunded schools and creating more rigorous academic standards, this law could help bridge the gap between high-performing and low-performing schools. What stands out to me is the emphasis on teacher training, which has been neglected for too long. Teachers play a pivotal role in shaping students’ futures, and improving the quality of teacher training could have a ripple effect on student outcomes. Furthermore, the law includes mechanisms for regularly reviewing and updating curricula to ensure they remain relevant in a rapidly changing world. However, I am curious about how this law will affect existing schools that are already meeting high standards.",
        "While I agree that improving education is essential, I have reservations about how effective this law will be in practice. One of my main concerns is the focus on standardized testing, which this law still emphasizes. In my opinion, standardized tests often fail to capture the full range of student abilities and can place undue stress on both students and teachers. Moreover, the additional resources provided to underperforming schools are welcomed, but I fear they may not be enough to make a real difference. We've seen similar policies in the past where funding was promised but either failed to materialize or was insufficient to tackle the systemic issues plaguing these schools. Additionally, I wonder if there's enough flexibility in this law to allow schools to innovate and adopt new teaching methods tailored to their students' needs.",
        "The city’s new traffic management plan has sparked a lot of debate, and for good reason. While I appreciate the intent behind it — reducing congestion, improving safety for pedestrians and cyclists, and lowering carbon emissions — I feel that it doesn’t fully address the city’s unique challenges. The addition of bike lanes is certainly a positive step, but the way they’re being implemented seems rushed and poorly planned. In some areas, bike lanes have replaced entire car lanes, leading to even more congestion during peak hours. There’s also the issue of enforcement. Without proper monitoring, cars often park illegally in bike lanes, defeating their purpose entirely. The plan could have been more effective if it focused more on public transportation and less on restricting car usage.",
        
        # Thread 3
        "The city's traffic management plan is flawed. Bike lanes are causing more congestion.",
        "While the bike lanes cause issues, they are a step toward making the city more bike-friendly.",
        "We need better planning and enforcement, but I agree this plan is a step in the right direction.",
        
        # Additional arguments
        "Public transportation should be improved instead of adding more bike lanes.",
        "Subsidizing small businesses to comply with environmental policies is essential.",
        "Teacher training reforms will have the most significant impact on education.",
        "I don’t think the policy will reduce emissions fast enough to meet climate goals.",
        "Banning cars from the city center is a more effective way to tackle congestion.",
        "Encouraging carpooling would also reduce traffic, not just more bike lanes.",
        "The education reform law should focus on reducing class sizes as well.",
        "Funding for renewable energy should be prioritized over short-term economic costs.",
        "Tax incentives for green technologies are a more practical solution.",
        "Road pricing would better address congestion than just adding bike lanes.",
        "More public awareness campaigns are needed to support these environmental policies."
    ]
}

df = pd.DataFrame(data)
df.head()

```

### **4. Using WIBA API**

#### **4.0. Creating Text Segments** (Optional)
The `create_segments` endpoint allows you to break long arguments into smaller parts, which is useful for applying argument detection and stance analysis on segments of a larger text.
```python

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


# Convert DataFrame to CSV format
csv_data = df.to_csv(index=False)

# Call the 'create_segments' API endpoint
create_segments_df = create_segments(csv_data, "text", window_size=3)

# Display the created segments
print(create_segments_df)
```

Example output:
```python
Created segments:
   id                                             text_segment
0   1  This is a very long argument that discusses multipl...
1   1  Then it discusses the economic costs of implementin...
2   1  Moreover, it argues about the long-term benefits of...
3   1  Finally, it touches on the importance of balancing ...
4   2  Another long post on education reform. It highlight...
5   2  It goes on to mention that teacher training is cruc...
6   2  Finally, it makes a case for increasing funding to ...
...
```

#### **4.1. Detecting Arguments**

We’ll first detect arguments in each of the posts using the `wiba_detect` function. The function sends the data to the WIBA API’s `/detect` endpoint and returns the detected arguments.

```python
def wiba_detect(df):
    url = 'http://wiba.dev/api/detect'
    csv_data = df.to_csv(index=False)
    payload = {"texts": csv_data}
    response = requests.post(url, json=payload)
    return pd.DataFrame(response.json())

# Detect arguments in the dataset
detect_df = wiba_detect(df)
print(detect_df)
```
Expected Output for the `detect` endpoint

```python
Detection results:
   id                                               text                                argument_prediction           argument_confidence
0   1  I strongly believe that this new policy will significantly reduce pollution...          Argument                    0.85
1   1  While the goal of reducing pollution is certainly commendable,...                       Argument                    0.75
2   1  Moreover, it argues about the long-term benefits of...                                  Argument                    0.90
3   1  The new law aimed at improving education outcomes is a much-needed ...                  Argument                    0.80
...
```

#### **4.2. Extracting Topics**

Next, we’ll use the `wiba_extract` function to identify the key topics being discussed in the posts.

```python
def wiba_extract(texts):
    url = 'http://wiba.dev/api/extract'
    payload = {"texts": texts}
    response = requests.post(url, json=payload)
    return pd.DataFrame(response.json())

# Extract topics from the detected posts
texts = detect_df['text'].tolist()
extract_df = wiba_extract(texts)
print(extract_df)
```

#### **4.3. Analyzing Stance**

Finally, we’ll analyze the stance of each post regarding the extracted topics using the `wiba_stance` function.

```python
def wiba_stance(texts, topics):
    url = 'http://wiba.dev/api/stance'
    payload = {"texts": texts, "topics": topics}
    response = requests.post(url, json=payload)
    return pd.DataFrame(response.json())

# Extract topics and analyze stance
extract_df = extract_df.rename(columns={"extracted_topic": "topic"})
topics = extract_df['topic'].tolist()
stance_df = wiba_stance(texts, topics)
print(stance_df)
```

#### **4.4. Combining Results**

We’ll combine the detected arguments, extracted topics, and stance results into a single DataFrame for further analysis.

```python
result_df = pd.concat([detect_df, extract_df, stance_df], axis=1)
print("Combined results:")
print(result_df)
```

---

### **5. Using `delibtools` for Deliberation Intensity Analysis**

The `delibtools` package allows us to measure the deliberation intensity of discussions in the dataset, focusing on argument clusters, argumentativeness, and deliberation depth.
```python
from delibtools import DeliberationIntensity, utils
```

#### **5.0. Initialize DeliberationIntensity Class**

```python
di = DeliberationIntensity(verbose=True)
```
Set verbose=True for outputting progress bars.

#### **5.1. Generating Thread IDs**
`The delibtools` package allows for finding and assigning posts that are found within the same tree-like structure of a reply-post network to a unique thread id.

```python
df['thread_id'] = utils.assign_reddit_threads(df)
```


#### **5.2. Calculating Deliberation Intensity**

We’ll calculate deliberation intensity on the dataset. The method computes argument density and cluster properties across the Reddit-style threads.

```python
# Prepare data for deliberation intensity analysis
result_df['argument'] = detect_df['argument_predictions'].map({'Argument': 1, 'NoArgument': 0})

# Calculate deliberation intensity
delib_intensity_df = di.calculate_deliberation_intensity(result_df)
print("Deliberation Intensity Results:")
print(delib_intensity_df)
```
Example Output
```python
   thread_id  d_cluster    d_arg  num_arguments  num_turns   sigma_1   sigma_2       dis
0          1   0.666667  0.352941              4          4  0.506842  0.493158  0.515114
1          5   0.666667  0.352941              3          3  0.472477  0.527523  0.514127
2          8   0.666667  0.352941              3          3  0.472477  0.527523  0.514127
3         11   0.666667  0.352941              3          3  0.472477  0.527523  0.514127
4         14   0.666667  0.352941              3          3  0.472477  0.527523  0.514127
```

#### **5.3. Visualizing Deliberation Intensity with ECDF**

We’ll visualize deliberation intensity using an empirical cumulative distribution function (ECDF) plot.

```python
# Plot ECDF to visualize deliberation intensity
di.plot_ecdf(delib_intensity_df)
```

---

### **6. Conclusion**

By leveraging the WIBA API and `delibtools`, we were able to detect arguments, extract key topics, analyze stances, and compute deliberation intensity from a sample dataset of Reddit-like posts.

This pipeline can be extended and customized for larger datasets, offering a powerful method for analyzing online discussions and argumentation in various contexts.

### **Further Reading and References**

[WIBA: What Is Being Argued? A Comprehensive Approach to Argument Mining](https://arxiv.org/abs/2405.00828)

[What Is Being Argued (WIBA)? An Application to Legislative Deliberation in the U.S. Congress](https://arxiv.org/abs/2407.06149)

[ArguSense: Argument-Centric Analysis of Online Discourse](https://arxiv.org/abs/2405.15930)


