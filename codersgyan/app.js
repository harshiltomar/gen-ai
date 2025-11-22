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
  //await saveToFile(content);
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
    temperature: "1",
    top_p: 0.2,
    stop: "11", // What exactly to force stop for
    max_completion_tokens: 1000, // Max tokens to generate
    frequency_penalty: 0,
    presence_penalty: 0,
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: `You are Jarvis! A smart assistant that can answer questions and help with tasks. Always consider the user's context and previous conversations when answering questions. Your task is to analyse given review and return the sentiment. Classify the review into positive, negative or neutral. Output must be single word along with a small sentence about the justification. You must return the output in valid JSON structure
        {
          "sentiment": "positive",
          "justification": "The food was good, but the service was slow. Overall, it was a good experience and my favorite dishes were the lasagna and the tiramisu."
        }
        `
      },
      {
        role: "user",
        content: "Review: The food was good, but the service was slow. Overall, it was a good experience and my favorite dishes were the lasagna and the tiramisu."
      },
    ],
    model: "openai/gpt-oss-20b",
  });
}

main();