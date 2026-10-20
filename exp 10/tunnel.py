import time
from pyngrok import ngrok

print("--- Starting Public Tunnel for Exam Score Prediction Dashboard ---")
try:
    # Open an HTTP tunnel on port 5000
    public_url = ngrok.connect(5000).public_url
    print("\n========================================================")
    print(f"🎉 Your App is LIVE on the Internet at:")
    print(f"👉 {public_url}")
    print("========================================================\n")
    print("Keep this terminal window open to keep the link active.")
    print("Press Ctrl + C to stop.")

    # Keep script running
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nTunnel stopped.")
    ngrok.kill()
except Exception as e:
    print("\nError starting tunnel:", e)
    print("Tip: If ngrok requires an auth token, create a free account at https://dashboard.ngrok.com/signup and run:")
    print("ngrok config add-authtoken YOUR_TOKEN")
