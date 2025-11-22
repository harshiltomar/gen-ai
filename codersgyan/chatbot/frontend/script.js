console.log("Welcome to Scira AI");

const input = document.querySelector("#input");
const chatContainer = document.querySelector("#chat-container");
console.log(input);

input?.addEventListener('keyup', handleEnter);

// Append message to UI and send it to the LLM and then append message to UI
function generate(text) {

    const msg = document.createElement('div');
    msg.className = 'my-6 bg-neutral-800 p-3 rounded-xl ml-auto max-w-fit';
    msg.textContent = text;

    chatContainer.appendChild(msg);
    input.value= ' ';
}

function handleEnter(e) {
    if(e.key === 'Enter') {
        const text = input?.value.trim();
        if(!text) {
            return;
        }

        generate(text);
    }
}
