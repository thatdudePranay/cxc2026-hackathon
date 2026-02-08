import os
import google.generativeai as genai

#genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
genai.configure(api_key="AIzaSyAPY30iRGHgKLqWbNie-kzL3OOmLVn9z6k")
model = genai.GenerativeModel('gemini-2.5-flash')

# Takes in user_speech, ocr_text and yolo_detections to determine if the specified target is nearby
def find_and_guide(user_speech, ocr_text, yolo_detections):
    ocr_string = ", ".join(ocr_text) if ocr_text else "No text detected"
    yolo_string = ", ".join(yolo_detections) if yolo_detections else "No objects detected"
    
    prompt = f"""
You are a navigation assistant helping a visually impaired person.

User said: "{user_speech}"

Nearby text (from OCR): {ocr_string}
Nearby objects (from camera): {yolo_string}

Task:
1. Figure out what the user is looking for from their speech
2. Check if it's visible in the OCR text or detected objects
3. Provide SHORT, CLEAR navigation instructions if found (e.g., "The Rexall is on your left, walk forward")
4. If not found, say "I don't see [target] nearby right now"

Keep instructions under 20 words, be direct and spatial (use left/right/forward/behind).

Respond in this EXACT format:
TARGET: [what they're looking for]
FOUND: [yes/no]
INSTRUCTIONS: [your guidance]
"""
    
    response = model.generate_content(prompt)
    text = response.text.strip()
    
    # Parse response
    lines = text.split('\n')
    target = ""
    found = False
    instructions = ""
    
    for line in lines:
        if line.startswith("TARGET:"):
            target = line.replace("TARGET:", "").strip()
        elif line.startswith("FOUND:"):
            found = "yes" in line.lower()
        elif line.startswith("INSTRUCTIONS:"):
            instructions = line.replace("INSTRUCTIONS:", "").strip()
    
    return {
        'target': target,
        'found': found,
        'instructions': instructions
    }

# Example usage:
if __name__ == "__main__":
    user_speech = "Hey guide where is the rexall?"
    ocr_text = ["Rexall Pharmacy - Right", "EXIT - Front - Far", "Shoppers Drug Mart - Right - Far"]
    yolo_detections = ["door", "sign", "person", "wall"]
    
    result = find_and_guide(user_speech, ocr_text, yolo_detections)
    
    print(f"Target: {result['target']}")
    print(f"Found: {result['found']}")
    print(f"Instructions: {result['instructions']}")