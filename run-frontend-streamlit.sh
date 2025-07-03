
# Make this file executable: chmod +x run-frontend-streamlit.sh
# Run it with: ./run-frontend-streamlit.sh

streamlit run app/frontend-streamlit/streamlit_app.py

print_blue "Opening browser tabs..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then

    explorer.exe "http://localhost:8501"
else

    xdg-open http://localhost:8501 >/dev/null 2>&1 &
fi

