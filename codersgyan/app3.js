import Groq from "groq-sdk";
import { tavily } from "@tavily/core";

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const tvly = tavily({ apiKey: process.env.TAVILY_API_KEY });


export async function main() {
  const chatCompletion = await getGroqChatCompletion();
  
  // Print the completion returned by the LLM.
  console.log(JSON.stringify(chatCompletion.choices[0].message, null, 2));
}

export async function getGroqChatCompletion() {
  const response = await groq.chat.completions.create({
    messages: [
      {
        role: "system",
        content: `You are Jarvis! A smart assistant that can answer questions and help with tasks. You have access to the following tools
        1. searchWeb({query}: {query: string}) //Search the latest information and real-time data from the internet`
      },
      {
        role: "user",
        content: "What is the latest iphone and what are it's specs"
      },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "websearch",
          description: "Search the latest information and real-time data from the internet",
          parameters: {
            type: "object",
            properties: {
              query: {
                type: "string",
                description: "The search query to search the internet for / to perform search on",
              },
            },
            required: ["query"]
          }
        }
      }
    ],
    tool_choice: 'auto',
    model: "openai/gpt-oss-20b",
  });

  const toolCalls = response.choices[0].tool_calls;

  if(!toolCalls) {
    console.log(`Assistant: ${response.choices[0].message.content}`);
    return response;
  }

  for (const tool of toolCalls) {
    console.log(`Tool: ${tool}`);
    const functionName = tool.function.name;
    const functionParams = tool.function.arguements;

    if(functionName === "websearch") {
      const result = await webSearch(JSON.parse(functionParams));
      console.log(`Result: ${result}`);
      return result;
    }
  }

  console.log(JSON.stringify(response.choices[0].message, null, 2));
  return response;
}

main();

async function webSearch({query}) {
    // API CALL to Tavily
    console.log("Calling WebSearch Tool...");
    console.log(`Query: ${query}`);

    const response = await tvly.search(query);  
    console.log(JSON.stringify(response, null, 2));
    return response;
}