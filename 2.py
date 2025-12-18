import google.generativeai as genai

genai.configure(api_key="AIzaSyAfrdlhsrN8DbKj68R7FNcsxqi15PKbA")

model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content("""
Write python code
return ONLY valid Python code.
No markdown.
No explanation.""")

code = response.text.strip().replace("```python", "").replace("```", "")

print(code)
