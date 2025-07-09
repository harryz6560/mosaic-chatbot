import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from 'react-markdown';
import {
  PlusIcon,
  SendIcon,
  UserIcon,
  BotIcon,
  ChevronDownIcon,
  ThumbsUpIcon,
  ThumbsDownIcon,
  CopyIcon,
  TrashIcon,
  ScanSearchIcon,
} from "lucide-react";
import { Input } from "./components/ui/input";
import './App.css';

const PremiumChatBotUI = () => {
  const [conversations, setConversations] = useState([]);
  const [selectedConversationId, setSelectedConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const messagesEndRef = useRef(null);
  const [isLoading, setIsLoading] = useState(false);
  const [feedback, setFeedback] = useState({});
  const initialized = useRef(false);
  const textareaRef = useRef(null);
  const [editingTitleId, setEditingTitleId] = useState(null);
  const [useDeepThink, setUseDeepThink] = useState(false);
  const fileInputRef = useRef(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);


  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = `${el.scrollHeight}px`;
    }
  }, [inputMessage]);

  useEffect(() => {
    const savedChats = JSON.parse(localStorage.getItem("chatHistory") || "[]");
    setConversations(savedChats.reverse());
  }, []);

  const handleSendMessage = async () => {
    if (inputMessage.trim() === "") return;

    const newUserMessage = { text: inputMessage, sender: "user", hidden: false};

    // Is this the first message in a new conversation?
    const isNewChat = messages.length === 0;

    const newMessages = [...messages, newUserMessage];
    setMessages(newMessages);
    setInputMessage("");
    setIsLoading(true);

    const fullHistory = newMessages.map((msg) => ({
      role: msg.sender === "user" ? "user" : "assistant",
      content: msg.text,
    }));

    try {
      const endpoint = useDeepThink ? '/api/deep_think' : '/api/chat';
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: inputMessage, history: fullHistory }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();
      const newBotMessage = { text: data.content, sender: "bot", hidden: false};
      const fullChat = [...newMessages, newBotMessage];
      setMessages(fullChat);
      setIsLoading(false);

      if (isNewChat) {
        const timestamp = new Date().toISOString();
        const summarizedTitle = inputMessage.length > 30
          ? inputMessage.slice(0, 30) + "..."
          : inputMessage;

        const newChat = {
          id: timestamp,
          title: summarizedTitle,
          messages: fullChat,
        };

        const savedChats = JSON.parse(localStorage.getItem("chatHistory") || "[]");
        const updatedChats = [...savedChats, newChat];

        localStorage.setItem("chatHistory", JSON.stringify(updatedChats));
        setConversations(updatedChats.reverse()); // Show in sidebar immediately
        setSelectedConversationId(timestamp);     // Highlight it right away
      } else {
        // If it's not a new chat, update messages in current chat in storage
        const updatedChats = conversations.map((chat) =>
          chat.id === selectedConversationId
            ? { ...chat, messages: fullChat }
            : chat
        );
        localStorage.setItem("chatHistory", JSON.stringify([...updatedChats].reverse()));
        setConversations(updatedChats);
      }

    } catch (error) {
      console.error("Error:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: `Error: ${error.message}`, sender: "bot" , hidden: false},
      ]);
      setIsLoading(false);
    }
  };


  async function sendStream() {
    const history = [...messages, { text: inputMessage, sender: "user", hidden: false }]
      .map(msg => ({
        role: msg.sender === "user" ? "user" : "assistant",
        content: msg.text
      }));
    setMessages(prev => [...prev, { sender: "user", text: inputMessage, hidden: false}]);
    setInputMessage("");
    setIsLoading(true);

    const resp = await fetch("/api/chat-stream", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({message: inputMessage, history})
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let botMsg = { sender: "bot", text: "", hidden: false};
    setMessages(prev => [...prev, botMsg]); 
    let msgIndex = messages.length;

    while(true) {
      const { value, done } = await reader.read();
      if(done) break;
      const chunk = decoder.decode(value);
      botMsg.text += chunk;
      setMessages(prev => {
        const arr = [...prev];
        arr[msgIndex] = botMsg;
        return arr;
      });
    }
    setIsLoading(false);
  }


  const handleFeedback = (messageIndex, isPositive) => {
    setFeedback(prev => ({
      ...prev,
      [messageIndex]: isPositive
    }));
    // Here you would typically send this feedback to your backend
    console.log(`Feedback for message ${messageIndex}: ${isPositive ? 'positive' : 'negative'}`);
  };

  const handleCopyMessage = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      // Optionally, you can show a brief notification that the text was copied
      console.log('Text copied to clipboard');
    }, (err) => {
      console.error('Could not copy text: ', err);
    });
  };

  const renderMessage = (message, index) => {

    if (message.hidden) return null;

    const isUser = message.sender === "user";

    return (
      <div
        key={index}
        className={`flex ${isUser ? "justify-end" : "justify-start"} animate-fadeIn`}
      >
        <div
          className={`max-w-2xl p-4 rounded-xl shadow ${
            isUser
              ? "bg-[#2a2b32] text-white"
              : "bg-[#444654] text-white"
          }`}
        >
          <div className="flex">
            {!isUser && (
              <BotIcon className="w-5 h-5 mr-3 mt-1 text-[#10A37F]" />
            )}
            <div className="markdown-content">
              <ReactMarkdown>{message.text}</ReactMarkdown>

              {!isUser && (
                <div className="mt-2 flex justify-end space-x-2">
                  <button
                    onClick={() => handleCopyMessage(message.text)}
                    className="p-1 rounded-full bg-[#3f414c] hover:bg-[#555770]"
                    title="Copy message"
                  >
                    <CopyIcon className="w-4 h-4 text-white" />
                  </button>
                  <button
                    onClick={() => handleFeedback(index, true)}
                    className={`p-1 rounded-full ${
                      feedback[index] === true ? "bg-green-500" : "bg-[#3f414c] hover:bg-[#4e4f58]"
                    }`}
                  >
                    <ThumbsUpIcon className="w-4 h-4 text-white" />
                  </button>
                  <button
                    onClick={() => handleFeedback(index, false)}
                    className={`p-1 rounded-full ${
                      feedback[index] === false ? "bg-red-500" : "bg-[#3f414c] hover:bg-[#4e4f58]"
                    }`}
                  >
                    <ThumbsDownIcon className="w-4 h-4 text-white" />
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Function to add messages with delay and simulate movement
  const initializeMessages = async () => {
    /*
    const cities = [
      {
        user: "What's the difference between let and const in JavaScript?",
        bot: "Here's a simple explanation:\n\n" +
             "- `let` allows you to reassign values\n" +
             "- `const` prevents reassignment\n\n" +
             "Example:\n" +
             "```javascript\n" +
             "let count = 1;\n" +
             "count = 2; // ✅ This works\n\n" +
             "const API_KEY = '123';\n" +
             "API_KEY = '456'; // ❌ This throws an error\n" +
             "```"
      }
    ];

    // Add each city with a delay
    for (let i = 0; i < cities.length; i++) {
      const city = cities[i];
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setMessages(prev => [
        ...prev,
        { sender: "user", text: city.user },
        { sender: "bot", text: city.bot }
      ]);
    } 
    */
  };

  const handleNewChat = () => {
    // Don't do anything if the current chat is empty
    if (messages.length === 0) {
      setMessages([]);
      setSelectedConversationId(null);
      return;
    }

    // Check if current chat is already saved
    const existingChat = conversations.find(chat =>
      chat.id === selectedConversationId &&
      JSON.stringify(chat.messages) === JSON.stringify(messages)
    );

    if (!existingChat) {
      const timestamp = new Date().toISOString();
      
      const newChat = {
        id: timestamp,
        title: messages[0]?.text.length > 30
          ? messages[0].text.slice(0, 30) + "..."
          : messages[0]?.text || "New Chat",
        messages,
      };

      const savedChats = JSON.parse(localStorage.getItem("chatHistory") || "[]");
      const updatedChats = [...savedChats, newChat];
      localStorage.setItem("chatHistory", JSON.stringify(updatedChats));
      setConversations(updatedChats.reverse());
    }

    // Reset for new chat
    setMessages([]);
    setSelectedConversationId(null);
  };

  const handleDeleteChat = (idToDelete) => {
    const updatedChats = conversations.filter(conv => conv.id !== idToDelete);
    setConversations(updatedChats);
    localStorage.setItem("chatHistory", JSON.stringify(updatedChats.reverse()));
    if (selectedConversationId === idToDelete) {
      setMessages([]);
      setSelectedConversationId(null);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    // Add placeholder to show it's uploading
    setUploadedFiles(prev => [
      ...prev,
      { name: file.name, status: "uploading" }
    ]);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Failed to upload file");

      const data = await res.json();

      // Update visual file state to "uploaded"
      setUploadedFiles(prev => 
        prev.map(f =>
          f.name === file.name ? { ...f, status: "uploaded" } : f
        )
      );

      // Inject as chat message
      setMessages(prev => [
        ...prev,
        { sender: "user", text: `Uploaded file content (${file.name}):\n\n${data.content}`, hidden: true }
      ]);

    } catch (err) {
      console.error("Upload failed:", err);
      setUploadedFiles(prev => 
        prev.map(f =>
          f.name === file.name ? { ...f, status: "error" } : f
        )
      );
      setMessages(prev => [
        ...prev,
        { sender: "bot", text: `Error reading ${file.name}: ${err.message}`, hidden: false}
      ]);
    }
  };



  // Call the initialization function when component mounts
  useEffect(() => {
    if (!initialized.current) {
      initialized.current = true;
      initializeMessages();
    }
  }, []);


  return (
  <div className="flex flex-col h-screen overflow-hidden bg-[#343541] text-[#ECECF1] font-sans text-sm">
    {/* Top Panel */}
    <div className="flex-none h-14 bg-[#111827] border-b border-gray-800 flex items-center px-4 shadow-md z-10">
      <img
        src="/petronas-logo.png"
        alt="Petronas Logo"
        className="h-8 w-8 object-contain mr-3"
      />
      <span className="text-white text-base font-semibold">Petronas Chat Assistant</span>
    </div>

    {/* Main Content */}
    <div className="flex flex-1 overflow-hidden">
      {/* Sidebar */}
      <div className="flex-none w-72 bg-[#202123] p-4 flex flex-col border-r border-[#2a2b32]">
        <button className="bg-[#10A37F] text-white rounded-full py-2 px-4 text-sm flex items-center justify-center mb-6 hover:bg-[#0e8a6b] transition-all duration-300 shadow"
          onClick={handleNewChat}
        >
          <PlusIcon className="w-4 h-4 mr-2" />
          <span className="font-semibold">New Chat</span>
        </button>
        <div className="flex-grow overflow-y-scroll space-y-2 custom-scrollbar">
          {conversations.length > 0 ? (
            conversations.map((conv) => (
              <div
                key={conv.id}
                onClick={() => {
                  if (selectedConversationId !== conv.id) {
                    setMessages(conv.messages);
                    setSelectedConversationId(conv.id);
                  }
                }}
                className={`relative py-2 px-3 cursor-pointer transition-all duration-200 flex items-center group ${
                  selectedConversationId === conv.id
                    ? 'bg-[#3f414c] text-white rounded-2xl shadow-inner'
                    : 'hover:bg-[#3f414c] rounded-2xl'
                }`}
              >
                <UserIcon
                  className="w-4 h-4 mr-2 text-gray-400 group-hover:text-[#10A37F]"
                />
                {editingTitleId === conv.id ? (
                  <input
                    type="text"
                    defaultValue={conv.title}
                    autoFocus
                    onBlur={(e) => {
                      const updatedChats = conversations.map((c) =>
                        c.id === conv.id ? { ...c, title: e.target.value } : c
                      );
                      setConversations(updatedChats);
                      localStorage.setItem("chatHistory", JSON.stringify(updatedChats.reverse()));
                      setEditingTitleId(null);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") e.target.blur();
                    }}
                    className="text-xs text-white bg-transparent border-none outline-none flex-grow"
                  />
                ) : (
                  <span
                    className="text-xs group-hover:text-white truncate flex-grow"
                    onDoubleClick={(e) => {
                      e.stopPropagation(); // Prevent triggering select while editing
                      setEditingTitleId(conv.id);
                    }}
                  >
                    {conv.title}
                  </span>
                )}

                <button
                  onClick={(e) => {
                    e.stopPropagation(); // Prevent triggering selection when deleting
                    handleDeleteChat(conv.id);
                  }}
                  className="ml-2 p-1 rounded hover:bg-red-600 transition-all"
                  title="Delete chat"
                >
                  <TrashIcon className="w-4 h-4 text-gray-400 hover:text-white" />
                </button>
              </div>
            ))
          ) : (
            <p className="text-xs text-gray-500 px-2">No chat history</p>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-[#343541]">
        {/* Scrollable Message Area */}
        <div className="flex-grow overflow-y-scroll px-4 py-6 custom-scrollbar">
          <div className="max-w-2xl mx-auto space-y-4">
            {messages.length === 0 ? (
              <div className="text-center text-gray-400 mt-10 animate-fadeIn space-y-4">
                <h2 className="text-2xl font-semibold text-white">👋 Welcome to Petronas Chat Assistant</h2>
                <p className="max-w-xl mx-auto text-sm text-gray-400">
                  I’m your intelligent assistant designed to support you with internal workflows, quick answers,
                  technical queries, and more.
                </p>
                <ul className="text-sm text-gray-400 list-disc list-inside max-w-md mx-auto text-left space-y-1">
                  <li>Ask about company policies, documents, or procedures</li>
                  <li>Get summaries from uploaded reports</li>
                  <li>Draft emails or meeting notes</li>
                  <li>Explain technical terms or code snippets</li>
                </ul>
                <p className="text-sm text-gray-500">Start by typing a message below ⬇️</p>
              </div>
            ) : (
              messages.map((message, index) => renderMessage(message, index))
            )}
            {isLoading && (
              <div className="flex justify-center items-center py-4">
                <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-[#10A37F]"></div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>
        
        {/* Input Area - ChatGPT-accurate Layout */}
        <div className="px-4 py-3 border-t border-[#2a2b32] bg-[#343541]">
          <div className="relative max-w-2xl mx-auto">
            <div className="rounded-xl border border-[#4b4b4f] bg-[#40414F] px-4 pt-3 pb-2 focus-within:ring-2 focus-within:ring-[#10A37F] flex flex-col space-y-3">

              {/* File preview inside input box */}
              {uploadedFiles.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {uploadedFiles.map((file, idx) => (
                    <div
                      key={idx}
                      className={`flex items-center space-x-2 px-3 py-1 rounded-lg text-sm font-medium
                        ${file.status === "uploaded" ? "bg-[#2d7a5f] text-white" :
                          file.status === "uploading" ? "bg-[#4b4b4f] text-gray-300" :
                          "bg-red-600 text-white"}
                      `}
                    >
                      <span className="truncate max-w-[150px]">{file.name}</span>
                      {file.status === "uploading" && <span className="text-xs animate-pulse">Uploading...</span>}
                      {file.status === "uploaded" && <span className="text-xs">✓</span>}
                      {file.status === "error" && <span className="text-xs">Failed</span>}
                    </div>
                  ))}
                </div>
              )}

              {/* Textarea on top */}
              <textarea
                ref={textareaRef}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                placeholder="Message ChatBot..."
                rows={1}
                className="w-full resize-none bg-transparent text-sm text-white placeholder-gray-400 focus:outline-none custom-scrollbar"
                style={{ maxHeight: "160px", overflowY: "auto", lineHeight: "1.5" }}
              />

              {/* Buttons row under textarea */}
              <div className="flex justify-between items-center">
                {/* Left: Add + Deep Think */}
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="p-1.5 rounded-full bg-[#3f414c] hover:bg-[#4e4f58] text-gray-300 hover:text-white transition"
                    title="Upload file"
                  >
                    <PlusIcon className="w-4 h-4" />
                  </button>

                  <input
                    type="file"
                    accept=".pdf,.txt,.csv,.xlsx"
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                  <button
                    onClick={() => setUseDeepThink(!useDeepThink)}
                    className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-all
                      ${useDeepThink
                        ? 'bg-[#10A37F] border-transparent text-white'
                        : 'bg-[#3f414c] border-[#4b4b4f] text-gray-300 hover:bg-[#4e4f58]'}`}
                    title="Toggle Deep Think Mode"
                  >
                    <ScanSearchIcon className="w-4 h-4 inline mr-1 -mt-0.5" />
                    Deep Think
                  </button>
                </div>

                {/* Right: Send */}
                <button
                  onClick={handleSendMessage}
                  className="p-2 rounded-full text-white hover:text-[#10A37F] transition"
                  title="Send message"
                >
                  <SendIcon className="w-5 h-5 transform rotate-45" />
                </button>
              </div>
            </div>

            {/* Tip below input */}
            <div className="text-xs text-gray-500 mt-3 text-center flex items-center justify-center">
              <ChevronDownIcon className="w-4 h-4 mr-1" />
              ChatBot may make mistakes. Please verify info about people, places, or facts.
            </div>
          </div>
        </div>
      </div>
    </div>
    {/* Footer */}
    <div className="text-center text-xs text-gray-500 py-3 border-t border-[#2a2b32] bg-[#202123]">
      Copyright © {new Date().getFullYear()}{" "}
      <a target="_blank" rel="noopener noreferrer" href="https://www.petronas.com/" className="text-[#10A37F] hover:underline">
        Petroliam Nasional Berhad (PETRONAS)
      </a>{" "}
      (20076-K),{" "}
      <a target="_blank" rel="noopener noreferrer" href="https://www.petronascanada.com/" className="text-[#10A37F] hover:underline">
        Petronas Energy Canada Ltd. (PECL)
      </a>, Development & Petroleum Engineering Team, Data Analytics (Unconventional Development) Program. All rights reserved.
    </div>
  </div>
  );
};

export default PremiumChatBotUI;