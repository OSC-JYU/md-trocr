
# md-trocr

An experimental TrOCR wrapper for https://huggingface.co/Kansallisarkisto/multicentury-htr-model
languages: Finnish, Swedish

## Docker (Recommended)

### GPU version

Build and run:

	make build
	make run

Or run interactively:

	make run-interactive

### CPU version (EXTREMELY SLOW!)

Build and run:

	make build-cpu
	make run-cpu

Or run interactively:

	make run-cpu-interactive

### Other Docker commands

	make help           # Show all available commands
	make logs           # View container logs
	make shell          # Access container shell
	make stop           # Stop container
	make clean          # Remove container and image

## Local installation and run (CPU version EXTREMELY SLOW!)

	python -m venv venv
	source venv/bin/activate
	pip install -r requirements_cpu.txt

run

	python api.py


## API

endpoint is http://localhost:9012/process

Payload is options json and file to be prosecced. 

Endpoint returns JSON with "store" URL, where one can download the result.

	{
	  "response": {
	    "type": "stored",
	    "uri": "/files/020b358c-8815-4bcb-9d08-287aa13532e0/text.txt"
	  }
	}


### Example API call 

Run these from MD-trocr directory:


Detect

	curl -X POST -H "Content-Type: multipart/form-data" \
	  -F "message=@test/message.json;type=application/json" \
	  -F "content=@test/htr3.polygons.json;type=application/json" \
	  -F "source=@test/htr3.jpg;type=image/jpeg" \
	  http://localhost:9012/process





