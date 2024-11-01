# Use a Node image to build the React app
FROM node:20-slim AS build

# Set working directory
WORKDIR /app

# Copy package.json and yarn.lock for dependencies
COPY ../package.json ../yarn.lock ./

# Install dependencies
RUN yarn install

# Copy the rest of the app source code
COPY .. .

# Build the app
RUN yarn build

# Use an Nginx image for serving the app
FROM nginx:alpine

# Copy the built files from the Node stage to Nginx
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port 3000
EXPOSE 3000

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
