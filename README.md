# StudyBuddyAI 🤖

An AI-powered study assistant that helps you learn from PDF documents by generating summaries, quizzes, and flashcards.

## Features

- 📚 PDF text extraction and processing
- 📝 AI-generated summaries
- ❓ Interactive multiple-choice questions
- 🎴 Flashcards for active recall
- 🧠 Concept memory tracking
- 📊 Progress monitoring

## Local Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/StudyBuddyAI.git
cd StudyBuddyAI
```

2. Create and activate a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

5. Run the application:
```bash
streamlit run app.py
```

## Deployment on Streamlit Cloud

1. Fork this repository to your GitHub account

2. Go to [Streamlit Cloud](https://streamlit.io/cloud)

3. Click "New app"

4. Select your forked repository

5. Set the main file path as `app.py`

6. Add your OpenAI API key in the secrets management section:
   - Go to the app's settings
   - Click on "Secrets"
   - Add your API key:
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```

7. Deploy!

## Environment Variables

The following environment variables are required:

- `OPENAI_API_KEY`: Your OpenAI API key

## Security Notes

- Never commit your `.env` file or API keys to the repository
- Use Streamlit Cloud's secrets management for deployment
- Keep your API keys secure and rotate them periodically

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 