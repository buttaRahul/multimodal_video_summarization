import { useState } from "react";
import DisplaySummary from "./DisplaySummary";
import InputBar from "./InputBar";
import { Container, Typography, Card, CardContent, Box } from "@mui/material";

const UrlHandler = () => {
  const [url, setUrl] = useState("");

  return (
    <Box 
      sx={{ 
        display: "flex", 
        justifyContent: "center", 
        alignItems: "center", 
        minHeight: "100vh",
        backgroundColor: "lightskyblue" 
      }}
    >
      <Container maxWidth="md">
        <Card 
          elevation={3} 
          sx={{ 
            p: 3, 
            borderRadius: 2, 
            width: "90%", 
            maxWidth: "700px", 
            mx: "auto",
            textAlign: "center",
            backgroundColor: "white"
          }}
        >
          <CardContent>
            <Typography 
              variant="h4" 
              gutterBottom 
              sx={{ fontWeight: "bold", textAlign: "center" }}
            >
              🎥 Video Summarization
            </Typography>
            <InputBar onSubmit={setUrl} />
            <DisplaySummary url={url} />
          </CardContent>
        </Card>
      </Container>
    </Box>
  );
};

export default UrlHandler;
