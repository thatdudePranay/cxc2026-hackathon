import os
import google.generativeai as genai

# {"Velocity: " + str(velocity) + "m/s" if velocity > 0 else ""}


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
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

Keep instructions under 20 words, be direct and spatial (use left/right/forward/behind/specific angles).
If what the user said is nonsensical respond with "Sorry I couldn't understand you."

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

    if not found and not instructions:
        instructions = f"Could not find the {target}. Please look around more"
    
    return {
        'target': target,
        'found': found,
        'instructions': instructions
    }

# Takes in vision.py warnings and creates a custom response
def generate_warning(detections):
    # Filter for critical objects that are close
    dangerous = [
        d for d in detections 
        if d['is_critical'] and d['distance'] < 3.0
    ]
    
    if not dangerous:
        return None
    
    # Sort by distance (closest first) and take the most dangerous
    dangerous.sort(key=lambda x: x['distance'])
    closest = dangerous[0]
    
    obj_name = closest['class_name']
    distance = closest['distance']
    direction = closest['direction']
    angle = closest['angle']
    
    prompt = f"""You are a guide, generate a SHORT urgent warning (MAX 5 WORDS) for a visually impaired person.

Object: {obj_name}
Distance: {distance}m
Direction: {direction}
Angle: {angle}° from center

Examples:
- "Car ahead two meters"
- "Person approaching from left"
- "Stop vehicle coming fast"
- "Bicycle on your right"

Warning (5 words max):"""
    
    response = model.generate_content(prompt)
    warning = response.text.strip()
    print(warning)
    return warning

# Generate intelligent critical warning from alert data
def generate_critical_warning(alert):
    """
    Takes a critical alert from vision.py and generates an intelligent,
    concise warning using Gemini.
    
    Args:
        alert: Dict with keys: severity, object, distance, direction, 
               is_approaching, velocity, message
    
    Returns:
        String with the generated warning (max 10 words)
    """
    try:
        obj_name = alert.get('object', 'object')
        distance = alert.get('distance', 0)
        direction = alert.get('direction', 'ahead')
        is_approaching = alert.get('is_approaching', False)
        velocity = alert.get('velocity', 0)
        
        prompt = f"""Generate an URGENT, SHORT warning (MAX 10 WORDS) for a visually impaired person.

Object: {obj_name}
Distance: {distance}m
Direction: {direction}
Approaching: {is_approaching}

Be CLEAR, DIRECT, and URGENT. Use spatial terms (left, right, ahead, behind).
Include the distance and action if needed (stop, watch out, caution). Keep in mind that this will be read by TTS,
so COMPLETELY avoid acronyms like m/s (instead say metres per second). 

Examples:
- "Stop! Car ahead three meters"
- "Caution! Person approaching from left"
- "Watch out! Truck right side two meters"

Warning (10 words max):"""
        
        response = model.generate_content(prompt)
        warning = response.text.strip()
        
        # Remove quotes if Gemini added them
        warning = warning.strip('"').strip("'")
        
        return warning
        
    except Exception as e:
        print(f"⚠️  Gemini warning generation failed: {e}")
        # Fallback to original message
        return alert.get('message', 'Danger detected')