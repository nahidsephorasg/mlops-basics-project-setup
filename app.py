# flask simple app with logging and custom exception handling
from flask import Flask, jsonify
import logging
from src.logger import LOG_FILE_PATH
from src.exception import CustomException
import sys

app = Flask(__name__)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


@app.route("/")
def home():
    logging.info("Home route accessed")
    return "Welcome to the MLOps Project!"


@app.route("/error")
def trigger_error():
    try:
        logging.info("Error route accessed, about to trigger an error")
        # Intentionally trigger a ZeroDivisionError
        result = 1 / 0
        return f"Result is {result}"
    except Exception as e:
        logging.error("An error occurred in /error route")
        raise CustomException(str(e), sys) from e


@app.errorhandler(CustomException)
def handle_custom_exception(error):
    response = {"error": str(error)}
    return jsonify(response), 500


if __name__ == "__main__":
    app.run(debug=True)
