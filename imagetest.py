import openai
import base64

openai.api_key = "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"

def process_image(image_path):
    # Open and read the image file
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Create OpenAI request to analyze the image
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "If the image is a stock chart than Analyze the stock chart and describe key insights such as trends, volatility, support and resistance levels, price patterns, volume, and significant price points."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]
    )
    
    # Extract the text response
    text_response = response.choices[0].message.content
    return text_response

# Example usage
image_path = "tesla.png"  # Replace with your image path
result = process_image(image_path)
print("AI Response:", result)
