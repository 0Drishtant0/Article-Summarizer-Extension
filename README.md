# Chrome Extension: Text Summarizer

![Extension Icon](/path/to/icon.png)

## Overview

This Chrome extension allows users to enter text in a popup and send it to a Flask server. The server utilizes the Hugging Face API for text summarization and returns the summarized text, which is then displayed in the extension.

## Features

- User-friendly interface with a text input field and a submit button.
- Utilizes the Hugging Face API for text summarization.
- Displays the summarized text in the extension.

## Usage

1. **Install the Extension:**
   - Clone or download this repository.
   - Open Chrome and navigate to `chrome://extensions/`.
   - Enable "Developer mode" in the top right corner.
   - Click "Load unpacked" and select the extension directory.

2. **Run the Flask Server:**
   - Install Flask on your system using 'pip install flask', or use your preferred package manager.
   - Run the Flask server using the terminal: `python -m flask run`.

3. **Use the Extension:**
   - Click on the extension icon in the Chrome toolbar.
   - Enter text in the popup and click the "Submit" button.
   - The extension will send the text to the Flask server, which uses the Hugging Face API for summarization.
   - The summarized text will be displayed in the extension popup.

## Configuration

- If running locally, update the Flask server URL in `popup.js` to match your local port's address (for flask server).
- Ensure the server has internet access to communicate with the Hugging Face API.

## Dependencies

- [Flask](https://flask.palletsprojects.com/): Web framework for the server.

## Credits

- This extension uses the Hugging Face API for text summarization. Visit [Hugging Face](https://huggingface.co/) for more information.

## License

This project is licensed under the [MIT License](LICENSE).

