"""Server entry point for OpenEnv multi-mode deployment."""
import uvicorn


def main():
    """Start the FastAPI server."""
    uvicorn.run("api.main:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
