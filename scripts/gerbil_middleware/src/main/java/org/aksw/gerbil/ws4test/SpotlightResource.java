package org.aksw.gerbil.ws4test;

import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;

import org.aksw.gerbil.transfer.nif.Document;
import org.aksw.gerbil.transfer.nif.Marking;
import org.aksw.gerbil.transfer.nif.TurtleNIFDocumentCreator;
import org.aksw.gerbil.transfer.nif.TurtleNIFDocumentParser;
import org.restlet.representation.Representation;
import org.restlet.resource.Post;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SpotlightResource extends ServerResource {

    private static final Logger LOGGER = LoggerFactory.getLogger(SpotlightResource.class);

    private TurtleNIFDocumentParser parser = new TurtleNIFDocumentParser();
    private TurtleNIFDocumentCreator creator = new TurtleNIFDocumentCreator();
    private SpotlightClient client = new SpotlightClient();

    @Post
    public String accept(Representation request) {
        Reader inputReader;
        try {
            inputReader = request.getReader();
        } catch (IOException e) {
            LOGGER.error("Exception while reading request.", e);
            return "";
        }
        Document document;
        try {
            document = parser.getDocumentFromNIFReader(inputReader);
        } catch (Exception e) {
            LOGGER.error("Exception while reading request.", e);
            return "";
        }
        LOGGER.debug("Request: " + document.toString());
        document.setMarkings(new ArrayList<Marking>(client.annotateSavely(document)));
        LOGGER.debug("Result: " + document.toString());
        String nifDocument = creator.getDocumentAsNIFString(document);
        return nifDocument;
    }
}
