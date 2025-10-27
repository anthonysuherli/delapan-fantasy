import { ImageResponse } from "next/server";

export const size = {
  width: 32,
  height: 32,
};

export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "linear-gradient(135deg, #0f172a, #1e293b)",
          color: "#38bdf8",
          fontSize: 18,
          fontWeight: 600,
          letterSpacing: 1,
        }}
      >
        DF
      </div>
    ),
    {
      ...size,
    }
  );
}
