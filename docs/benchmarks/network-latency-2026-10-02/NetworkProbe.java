import java.io.*;
import java.net.*;
import java.util.*;

/** Standalone Android shell probe. Does not install or modify any app. */
public class NetworkProbe {
    public static void main(String[] args) throws Exception {
        String host = args[0];
        int port = Integer.parseInt(args[1]);
        int count = args.length > 2 ? Integer.parseInt(args[2]) : 40;
        byte[] payload = new byte[1024 * 1024];
        new Random(42).nextBytes(payload);
        byte[] response = new byte[payload.length];
        int[][] sizes = {{64,64}, {64*1024,64*1024}, {128*1024,128*1024},
                         {256*1024,256*1024}, {512*1024,512*1024}};
        long t = System.nanoTime();
        try (Socket socket = new Socket()) {
            socket.setTcpNoDelay(true);
            socket.connect(new InetSocketAddress(host,port),5000);
            socket.setSoTimeout(10000);
            System.out.printf(Locale.US,"connect_ms=%.3f%n",(System.nanoTime()-t)/1e6);
            DataOutputStream out = new DataOutputStream(socket.getOutputStream());
            DataInputStream in = new DataInputStream(socket.getInputStream());
            System.out.println("upload_bytes,download_bytes,sample,roundtrip_ms");
            for (int[] size : sizes) {
                for (int i=-5;i<count;i++) {
                    long start=System.nanoTime();
                    out.writeInt(size[0]); out.writeInt(size[1]);
                    out.write(payload,0,size[0]); out.flush();
                    in.readFully(response,0,size[1]);
                    double elapsed=(System.nanoTime()-start)/1e6;
                    for(int j=0;j<size[1];j++) if(response[j]!=payload[j])
                        throw new IOException("Response mismatch");
                    if(i>=0) System.out.printf(Locale.US,"%d,%d,%d,%.3f%n",size[0],size[1],i,elapsed);
                    Thread.sleep(30);
                }
            }
        }
    }
}
