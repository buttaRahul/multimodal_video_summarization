import { useState } from "react";
import { Container, TextField, Button, Typography, Box, Card } from "@mui/material";

const InputBar = ({ onSubmit }) => {
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = () => {
    if (!url) {
      setError("Please enter a valid URL");
      return;
    }
    setError("");
    onSubmit(url);
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 4 }}>
      <Card elevation={3} sx={{ p: 3, borderRadius: 2 }}>
        <Box display="flex" alignItems="center" gap={2}>
          <TextField
            label="Enter Video URL"
            variant="outlined"
            fullWidth
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            sx={{ borderRadius: 2 }}
          />
          <Button
            variant="contained"
            color="primary"
            onClick={handleSubmit}
            sx={{
              whiteSpace: "nowrap",
              fontWeight: "bold",
              p: 1.5,
              textTransform: "none",
              boxShadow: "none",
            }}
          >
            🚀 Submit
          </Button>
        </Box>
        {error && (
          <Typography color="error" sx={{ mt: 1, textAlign: "center", fontWeight: "bold" }}>
            {error}
          </Typography>
        )}
      </Card>
    </Container>
  );
};

export default InputBar;
