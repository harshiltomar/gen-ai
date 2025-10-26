import Groq from "groq-sdk";
import fs from "fs/promises";
import path from "path";

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

export async function main() {
  const chatCompletion = await getGroqChatCompletion();
  const content = chatCompletion.choices[0]?.message?.content || "";
  
  // Print the completion returned by the LLM.
  console.log(content);
  
  // Save to .md file
  await saveToFile(content);
}

async function saveToFile(content) {
  try {
    // Generate filename with timestamp
    const timestamp = new Date().toISOString().replace(/:/g, '-').split('.')[0];
    const filename = `response-${timestamp}.md`;
    
    // Write to file
    await fs.writeFile(filename, content, 'utf-8');
    console.log(`\nResponse saved to: ${filename}`);
  } catch (error) {
    console.error('Error saving file:', error);
  }
}

export async function getGroqChatCompletion() {
  return groq.chat.completions.create({
    messages: [
      {
        role: "user",
        content: "Explain the importance of fast language models",
      },
    ],
    model: "openai/gpt-oss-20b",
  });
}

main();