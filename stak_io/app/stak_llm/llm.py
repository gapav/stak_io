from openai import OpenAI

class SkiingCoachLLM:
    def __init__(self, api_key: str):
        """
        Initialize the OpenAI client with the provided API key.
        """
        self.client = OpenAI(api_key=api_key)

    def generate_feedback(self, max_angle: float) -> str:
        """
        Generate skiing feedback based on the maximum hip angle.

        Args:
            max_angle (float): The maximum hip angle found in the analysis.

        Returns:
            str: Feedback generated by the LLM or an error message if it fails.
        """
        prompt = (
            f"The maximum hip angle in the skiing video is {max_angle:.2f}°.\n"
            "Provide detailed feedback on how this might impact the user's skiing technique "
            "and suggest possible improvements."
        )

        try:
            # Use the OpenAI client to create a chat completion
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional skiing coach."},
                    {"role": "user", "content": prompt},
                ],
            )
            # Extract and return the generated content
            return response.choices[0].message.content
        except Exception as e:
            # Catch and return error messages for debugging
            return f"An error occurred while generating feedback: {e}"
