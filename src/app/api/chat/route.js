import {NextResponse} from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import { HfInference } from '@huggingface/inference'
import OpenAI from 'openai'

const systemPrompt = `You are an AI assistant for a 'Rate My Professor' platform, designed to help students find the best professors based on their specific queries. Your primary function is to analyze each student's request, retrieve relevant professor information using a Retrieval-Augmented Generation (RAG) system, and provide details about the top 3 professors who best match the student's needs.

For each query:

Analyze the student's request to understand their specific needs, preferences, and any criteria they’ve mentioned.
Retrieve information about the top 3 professors who align with the student's query using the RAG system.
Present the information clearly, concisely, and in an organized manner.
For each professor, provide:

Name and title
Department or field of expertise
Notable courses taught
Brief summary of teaching style, strengths, and any relevant feedback
Overall rating (if available)
Other relevant details that address the student’s specific query
Guidelines:

Be Objective: Present factual information without bias or unnecessary comparisons unless explicitly asked by the student.
Relate to the Query: Highlight how each professor’s qualities specifically relate to the student’s request.
Ask for Clarification: If the query is too broad, vague, or lacks detail, ask the student for more information to refine the results.
Handle Limitations Gracefully: If there's insufficient information to fully answer a query, inform the student and suggest how they might refine their search.
Your responses should be professional, respectful, and tailored to each student's unique needs, ensuring that they receive the most relevant and helpful information possible.`

const hf = new HfInference(process.env.HF_TOKEN)

const POST = async(req) =>{
    const data = await req.json()
    const text = data[data.length - 1].content

    const pinecone = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY
    })

    const pineConeIndex = pinecone.index('rag').namespace('professor-reviews')

    const embedding = await hf.featureExtraction({
        model: "intfloat/multilingual-e5-large",
        inputs: text
    })

    const queriedResults = await pineConeIndex.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding
    })

    let resultString = ''
    queriedResults.matches.forEach((match)=>{
        resultString += `
        Returned Results:
        Professor: ${match.id}
        Review: ${match.metadata.review}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n`
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder();
            try {
                for await (const chunk of hf.chatCompletionStream({
                    model: "mistralai/Mistral-7B-Instruct-v0.2",
                    messages: [
                        {role: 'system', content: systemPrompt},
                        ...lastDataWithoutLastMessage,
                        {role: 'user', content: lastMessageContent},
                    ],
                    max_tokens: 250,
                    temperature: 0.2,
                })) {
                    if (chunk.choices && chunk.choices.length > 0) {
                        const content = chunk.choices[0].delta.content;
                        if(content){
                            const text = encoder.encode(content);
                            controller.enqueue(text);
                        }
                        
                    }
                }
            } catch (err) {
                controller.error(err);
            } finally {
                controller.close();
            }
        },
    });

    return new NextResponse(stream);

}