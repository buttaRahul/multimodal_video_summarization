import { useState, useEffect } from "react";
import { Container, Card, CardContent, Typography, CircularProgress, Fade, Box } from "@mui/material";
import { styled } from "@mui/system";

const ScrollContainer = styled("div")({
  display: "flex",
  overflowX: "auto",
  gap: "12px",
  padding: "10px",
  scrollbarWidth: "thin",
  "&::-webkit-scrollbar": {
    height: "8px",
  },
  "&::-webkit-scrollbar-thumb": {
    backgroundColor: "#ccc",
    borderRadius: "4px",
  },
});

const FrameImage = styled("img")({
  height: "150px", 
  borderRadius: "8px",
  flexShrink: 0,
});

const DisplaySummary = ({ url }) => {
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [frames, setFrames] = useState([]);

  useEffect(() => {
    if (!url) return;

    setLoading(true);
    setError(""); 
    setSummary(""); 
    setFrames([]);

    fetch("http://localhost:8000/submit-url/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to fetch data.");
        }
        return response.json();
      })
      .then((data) => {
        setSummary(data.summary_response);
        setFrames(data.frames);
      })
      .catch(() => setError("Failed to fetch data. Please try again."))
      .finally(() => setLoading(false));
  }, [url]);

  return (
    <Container maxWidth="md" sx={{ mt: 4, bgcolor: "#f9f9f9", p: 3, borderRadius: 2 }}>
      {loading && <CircularProgress sx={{ display: "block", mx: "auto", my: 2 }} />}
      {error && (
        <Typography color="error" sx={{ mt: 2, textAlign: "center", fontWeight: "bold" }}>
          {error}
        </Typography>
      )}

      {frames.length > 0 && (
        <Box sx={{ bgcolor: "white", borderRadius: 2, p: 2, boxShadow: 1, mb: 3 }}>
          <Typography variant="h6" sx={{ fontWeight: "bold", color: "primary.main", mb: 2 }}>
            🖼️ Key Frames
          </Typography>
          <ScrollContainer>
            {frames.map((frame, index) => (
              <FrameImage
                key={index}
                src={`data:image/jpeg;base64,${frame}`}
                alt={`Frame ${index + 1}`}
                loading="lazy"
              />
            ))}
          </ScrollContainer>
        </Box>
      )}

      {summary && (
        <Fade in={Boolean(summary)} timeout={500}>
          <Card elevation={3} sx={{ borderRadius: 2, p: 2, bgcolor: "white" }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: "bold", color: "primary.main" }}>
                📜 Summary
              </Typography>
              <Typography variant="body1" sx={{ mt: 1, color: "text.secondary" }}>
                {summary}
              </Typography>
            </CardContent>
          </Card>
        </Fade>
      )}
    </Container>
  );
};

export default DisplaySummary;
