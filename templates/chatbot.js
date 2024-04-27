function sendMessage() {
    var userInput = document.getElementById("userInput").value;
    var chatWindow = document.getElementById("chatWindow");

    // Create a new message element
    var messageElement = document.createElement("div");
    messageElement.classList.add("message");
    messageElement.textContent = userInput;

    // Append the message element to the chat window
    chatWindow.appendChild(messageElement);

    // Clear the input field after sending the message
    document.getElementById("userInput").value = "";
}
